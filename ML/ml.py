#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
#from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
import numpy as np

def DeepDenseNN(features):
    input_img = Input(shape=(features,))
    #print(input_img)
    #input_img = Dense(128, activation='relu', kernel_initializer='he_uniform', input_shape=(features,))
    layer_1 = Dense(64, activation='linear', kernel_initializer='he_uniform' )(input_img)
    layer_2 = Dense(32, activation='tanh', kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(16, activation='relu', kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(8, activation='tanh', kernel_initializer='he_uniform' )(layer_3)
    layer_5 = Dense(4, activation='elu', kernel_initializer='he_uniform' )(layer_4)
    Label = Dense(1, activation='sigmoid', kernel_initializer='he_uniform')(layer_5)    
    model = Model(input_img, Label)
    return model

def SaveModel(model, path_model, path_weights):
    model_json = model.to_json()
    with open(path_model, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path_weights)
    print("Model is saved to disk\n")

def LoadModel(path_model, path_weights, optimizer, loss):
    from keras.models import model_from_json
    json_file = open(path_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    print("Model is loaded from disk\n")
    loaded_model.compile(optimizer=optimizer, loss=loss)
    return loaded_model

def NN(train,label,test_size,validation_split,batch_size,num_ep,optimizer,loss,output_path_predict,output_path_mod,output_path_weight):
	X_train, X_test, y_train, y_test = train_test_split(train, label, 
														test_size = test_size, random_state = 56) #0.4
	#print(X_train, X_test, y_train, y_test)
	#batch_size = 1024
	#num_ep = 15
	features = train.shape[1]
	print(features)
	model = DeepDenseNN(features)

	model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

	model.fit(X_train, y_train,
			epochs=num_ep,
			batch_size=batch_size,
			validation_split=validation_split) #0.25

	model.summary()
	Class = model.predict(X_test, batch_size)
	
	Class = np.array(Class)
	
	g=open(output_path_predict,'w')
	Class.tofile(g,"\n")
	g.close()
	
	count = 0
	Tq, Fq, Ts, Fs = 0,0,0,0
	for i in range(y_test.shape[0]):
		#print(Tq,Fq,Ts,Fs)
		if(Class[i]<0.5):
			Class[i] = 0
		if(Class[i]>=0.5):
			Class[i] = 1
		if(Class[i]==y_test[i]):
			count+=1
		if(Class[i]==1):
			if(Class[i]==y_test[i]):
				Tq += 1
			else:
				Fq += 1
		if(Class[i]==0):
			if(Class[i]==y_test[i]):
				Ts += 1
			else:
				Fs += 1
	print("Accuracy:",              count/y_test.shape[0])
	print("AGN precision:",     Tq/(Tq+Fq))
	print("nonAGN precision:",    Ts/(Ts+Fs))
	print("AGN completness:",       Tq/(Tq+Fs))
	print("nonAGN completness:",     Ts/(Ts+Fq))
	print("AGN_F:",     2*(Tq/(Tq+Fq)*Tq/(Tq+Fs))/(Tq/(Tq+Fq)+Tq/(Tq+Fs)) )
	print("non_AGN_F:",     2*(Ts/(Ts+Fs)*Ts/(Ts+Fq))/(Ts/(Ts+Fq)+Ts/(Ts+Fs)) )
	SaveModel(model,output_path_mod,output_path_weight)