import os
import librosa
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential,regularizers
from tensorflow.keras.layers import Conv2D,Flatten,Dense,BatchNormalization,Dropout,MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pydub 
import csv

commands_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

#This block is used for making dataset for silence as it was not provided.
# So this block breaks audios from background_noises into 1 sec fragments.
def silence_dataset():
    noise_path = "train/train/audio/_background_noise_"
    noise_files = os.listdir(noise_path)
    if(os.path.exists(noise_path+'/silence')):
        pass
    else:
        os.mkdir(noise_path+'/silence')
    for audio_file in noise_files:
        audio = pydub.AudioSegment.from_file(noise_path+'/'+audio_file,format = 'wav')
        for i, sil in enumerate(audio[::1000]):
            with open(noise_path+"/silence/sil-{}-{}.wav".format(audio_file,i), "wb") as f:
                sil.export(f, format="wav")

# silence()

#This block generates MFCC's for all the audio files and store them in a json file.
#I chose json because it is fast and handy.
def build_dataset():
    mfcc = []
    audio_details = {'command' : [],'filename' : [],'mfcc' : []}
    audio_path = 'train/train/audio'
    folder_names = os.listdir(audio_path)

    for folder in folder_names:
        command = ""
        if(folder=="_background_noise_"):
            command = "silence"
            audio_path = 'train/train/audio'+'/'+folder    
            audios = os.listdir(audio_path)
        else:
            command = folder
        audios = os.listdir(audio_path+'/'+command)
        for audio in audios:
            if(librosa.get_duration(filename=audio_path+'/'+command+'/'+audio)==1.0):  #checks if the audio file is exactly of 1 sec or not
                signal, sr = librosa.load(audio_path+'/'+command+'/'+audio)
                mfcc = librosa.feature.mfcc(signal, sr, n_mfcc = 13)
                audio_details['mfcc'].append(mfcc.tolist())
                audio_details['filename'].append(audio)
                if(command not in commands_list):
                    audio_details['command'].append(commands_list.index('unknown'))
                else:
                    audio_details['command'].append(commands_list.index(command))
        print(folder)

    with open('train/train/mfcc_kaggle.json','w') as f:
        json.dump(audio_details,f,indent=2)

#Loads dataset from json file
def load_dataset():
    if(os.path.exists('train/train/mfcc_kaggle.json')):
        with open('train/train/mfcc_kaggle.json','r') as f:
            audio_details = json.load(f)
            
        X = np.array(audio_details['mfcc'])
        Y = np.array(audio_details['command'])

        Y = to_categorical(Y)

        #SPLIT training and testing DATASET
        X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2) 
        
    else:
        build_dataset()
        load_dataset()

    #Converting data from 2D to 3D because Conv2D takes input in 3D.
    X_train = X_train[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    return X_train,X_test,Y_train,Y_test

# Convolutional Neural Network
def neural_nework():
    X_train,X_test,Y_train,Y_test = load_dataset()
    if(os.path.exists('model.json')):       #if saved model already exists
        with open("model.json","r") as f: 
            json_model = json.load(f)
        model = tf.keras.models.model_from_json(json_model)
        model.load_weights('Best_weights.hdf5')
        adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ["accuracy"])
        loss,acc = model.evaluate(X_test,Y_test) 
        print("loss=",loss,"acc = ",acc)

    else:        
        model = Sequential()
        model.add(Conv2D(64,(6,6),activation = 'relu',input_shape = (X_train[0].shape),kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Conv2D(128,(6,6), activation = 'relu',kernel_regularizer=regularizers.l2(0.0001)))
        model.add(BatchNormalization()) 
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(Dropout(0.2)) 

        model.add(Flatten())
        model.add(Dense(32, activation = 'relu',kernel_regularizer=regularizers.l2(0.0001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(len(commands_list), activation = 'softmax'))

    #This function changes learning rate from 0.001 to 0.0001 after 15 epochs.
        def scheduler(epoch, lr):
            if epoch < 15:
                return lr
            else:
                return lr * 0.1
        learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
        adam = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ["accuracy"])
        #Saves Best Weights Only
        checkpoint = tf.keras.callbacks.ModelCheckpoint("Best_weights.hdf5",monitor = 'val_accuracy',save_best_only = True, mode = 'max')
        model.fit(X_train,Y_train, batch_size = 50 ,epochs = 30,shuffle=True, validation_data = (X_test,Y_test),callbacks=[checkpoint,learning_rate])
        model_json = model.to_json()

        with open('model.json','w') as f:
            json.dump(model_json,f)
    return model


#Takes testing audio one at a time generate its MFCC, use the model to predict and returns the classfied command. 
def testing_outputs(filepath, model):
    
    signal, sr = librosa.load(filepath)
    mfcc = librosa.feature.mfcc(signal, sr, n_mfcc = 13)
    mfcc = np.array(mfcc)
    mfcc = mfcc[np.newaxis,...,np.newaxis]
    preds = model.predict(mfcc)
    return commands_list[preds.argmax()]

def main():
    final_outputs = [] 
    model = neural_nework()
    test_audio_path = "test/test/audio"
    test_audios = os.listdir(test_audio_path)
    for test_audio in test_audios:              #for all testing audio files
        filepath = test_audio_path+'/'+test_audio
        out = testing_outputs(filepath=filepath,model=model)
        filename = filepath.split('/')[-1]
        final_outputs.append([filename,out])
        print(filename)
    with open('kaggle_outputs.csv','w',newline='\r') as f:
        writer = csv.writer(f)
        writer.writerow(["fname","lable"])
        writer.writerows(final_outputs)            

main()
    