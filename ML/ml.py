#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
#from keras.layers.normalization import BatchNormalization
import pandas as pd
import os
import math

from keras.layers import Input, Dense, Dropout
from keras.layers import experimental, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, add
from keras.models import Model
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
import tensorflow as tf
from keras import backend as K
import numpy as np

import sklearn.metrics as skmetrics

def ImageDeepDenseNN(input_shape):
    inputs = Input(shape=input_shape)
    # Image augmentation block
    

    # Entry block
    x = experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = Conv2D(32, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)

def DeepDenseNN(features):
    input_img = Input(shape=(features,))
    #print(input_img)
    #input_img = Dense(128, activation='relu', kernel_initializer='he_uniform', input_shape=(features,))
    #layer_1 = Dense(64, activation='relu', kernel_initializer='he_uniform' )(input_img)
    layer_1 = Dense(64, activation='linear', kernel_initializer='he_uniform' )(input_img)
    #layer_2 = Dense(32, activation='relu', kernel_initializer='he_uniform' )(layer_1)
    layer_2 = Dense(32, activation='tanh', kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(16, activation='relu', kernel_initializer='he_uniform' )(layer_2)
    #layer_4 = Dense(8, activation='relu', kernel_initializer='he_uniform' )(layer_3)
    layer_4 = Dense(8, activation='tanh', kernel_initializer='he_uniform' )(layer_3)
    #layer_5 = Dense(4, activation='relu', kernel_initializer='he_uniform' )(layer_4)
    layer_5 = Dense(4, activation='elu', kernel_initializer='he_uniform' )(layer_4)
    #layer_10 = Dropout(.5, input_shape=(features,))(layer_9)
    #layer_6 = Dropout(.1, input_shape=(features,))(layer_5)
    Label = Dense(1,activation="sigmoid", kernel_initializer='he_uniform' )(layer_5)
    model = Model(input_img, Label)
    return model
def DeepDark_DenseNN(features):
    input_img = Input(shape=(features,))
    layer_1 = Dense(8, activation='linear', kernel_initializer='he_uniform' )(input_img)
    layer_2 = Dense(16, activation='relu', kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(32, activation='tanh', kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(64, activation='relu', kernel_initializer='he_uniform' )(layer_3)
    layer_5 = Dense(32, activation='tanh', kernel_initializer='he_uniform' )(layer_4)
    layer_6 = Dense(16, activation='relu', kernel_initializer='he_uniform' )(layer_5)
    layer_7 = Dense(8, activation='tanh', kernel_initializer='he_uniform' )(layer_6)
    layer_8 = Dense(4, activation='elu', kernel_initializer='he_uniform' )(layer_7)
    Label = Dense(1,activation="sigmoid", kernel_initializer='he_uniform' )(layer_8)
    model = Model(input_img, Label)
    return model

def DeepDarkDenseNN(features):
    input_img = Input(shape=(features,))
    layer_1 = Dense(32, activation='linear', kernel_initializer='he_uniform' )(input_img)
    layer_2 = Dense(64, activation='relu', kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(32, activation='tanh', kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(16, activation='relu', kernel_initializer='he_uniform' )(layer_3)
    layer_5 = Dense(8, activation='tanh', kernel_initializer='he_uniform' )(layer_4)
    layer_6 = Dense(4, activation='elu', kernel_initializer='he_uniform' )(layer_5)
    Label = Dense(1,activation="sigmoid", kernel_initializer='he_uniform' )(layer_6)
    model = Model(input_img, Label)
    return model

def NoDeepDenseNN(features):
    input_img = Input(shape=(features,))
    Label = Dense(1, activation='sigmoid', kernel_initializer='he_uniform' )(input_img)
    #layer_2 = Dense(features/2, activation='tanh', kernel_initializer='he_uniform' )(input_img)
    #Label = Dense(1, activation="sigmoid", kernel_initializer='he_uniform' )(layer_1)
    return Model(input_img, Label)
    

def SaveModel(model, path_model, path_weights, name):
    model_json = model.to_json()
    with open(f"{path_model}_{name}", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"{path_weights}_{name}")
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

def PreDelete(origin_path,main_name,trash_name):
	num_skipped = 0
	for folder_name in (main_name, trash_name):
		folder_path = f"{origin_path}/{folder_name}"
		for fname in os.listdir(folder_path):
			fpath = f"{folder_path}/{fname}"
			try:
				fobj = open(fpath, "rb")
				is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
			finally:
				fobj.close()

			if not is_jfif:
				num_skipped += 1
				# Delete corrupted image
				os.remove(fpath)
	print("Deleted %d images" % num_skipped)

def PreProcesing(image_size,batch_size,origin_path):

	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		origin_path,
		validation_split=0.2,
		subset="training",
		seed=1337,
		#color_mode="grayscale",
		#label_mode="binary",
		image_size=image_size,
		batch_size=batch_size
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		origin_path,
		validation_split=0.2,
		subset="validation",
		seed=1337,
		#color_mode="grayscale",
		#label_mode="binary",
		image_size=image_size,
		batch_size=batch_size
	)
	'''
	import matplotlib.pyplot as plt

	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(int(labels[i]))
			plt.axis("off")
	'''
	#print(len(train_ds), len(val_ds))
	train_ds = train_ds.prefetch(buffer_size=32)
	val_ds = val_ds.prefetch(buffer_size=32)
	#print(train_ds)
	return train_ds, val_ds

def ImageNN(image_size,batch_size,train_ds,val_ds,epochs,path_model,path_weights):
	print(val_ds)
	model = ImageDeepDenseNN(input_shape=image_size + (3,))
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
	model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=batch_size)
	model.summary()
	SaveModel(model,path_model,path_weights)

def TestNN(test_path,image_size,batch_size,path_model,path_weights):
	model = LoadModel(path_model,path_weights,optimizer="adam", loss="binary_crossentropy")
	test_sample = tf.keras.preprocessing.image_dataset_from_directory(
		test_path,
		image_size=image_size,
		batch_size=batch_size
	)
	predict = model.predict(test_sample, batch_size)
	print(predict)
	count = len(predict)
	one = 0
	for i in range(count):
		if(predict[i] > 0.5):
			one += 1
	print(f"Accuracy of test sample:	{(one/count)*100}%")

def Test_one():
	from tensorflow import keras
	name = "PS1_z"
	path_model = f"/home/kiril/github/ML_with_AGN/ML/models/model_img_{name}"
	path_weights = f"/home/kiril/github/ML_with_AGN/ML/models/weights_img_{name}"
	
	model = LoadModel(path_model,path_weights,optimizer="adam", loss="binary_crossentropy")
	img = keras.preprocessing.image.load_img(
		"/home/kiril/github/ML_data/GALAXY/image/jpg/PS1_z/164.127958333333_57.787918055555295.png"
	)
	print(img)
	#img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img, 0)  # Create batch axis

	predictions = model.predict(img_array)
	score = predictions[0]
	print(
		"This image is %.2f percent AGN and %.2f percent GAL."
		% (100 * (1 - score), 100 * score)
	)

def Image_ML(origin_path, image_size, batch_size, epochs, test_path, path_model, path_weights):
	train_ds, val_ds = PreProcesing(image_size,batch_size,origin_path)
	ImageNN(image_size,batch_size,train_ds,val_ds,epochs,path_model,path_weights)
	TestNN(test_path,image_size,batch_size,path_model,path_weights)
def TEST_IMG():
	image_size = (100, 100)
	batch_size = 32
	main_path = "/home/kiril/github/ML_data/test"

	for name in os.listdir(main_path):
		path_model = f"/home/kiril/github/ML_with_AGN/ML/models/model_img_{name}"
		path_weights = f"/home/kiril/github/ML_with_AGN/ML/models/weights_img_{name}"
		for name_ in [ "test_1", "test_2" ]:
			test_path = f"/home/kiril/github/ML_data/{name_}/{name}"
			TestNN(test_path,image_size,batch_size,path_model,path_weights)
			
def Start_IMG():
	image_size = (100, 100)
	batch_size = 32
	epochs = 5
	main_path = "/home/kiril/github/ML_data/test"
	
	for name in os.listdir(main_path):
		origin_path = f"{main_path}/{name}"
		test_path = f"/home/kiril/github/ML_data/test_1/{name}"
		path_model = f"/home/kiril/github/ML_with_AGN/ML/models/model_img_{name}"
		path_weights = f"/home/kiril/github/ML_with_AGN/ML/models/weights_img_{name}"
		_name = os.listdir(origin_path)
		PreDelete(origin_path,_name[0],_name[1])
		Image_ML(origin_path, image_size, batch_size, epochs, test_path, path_model, path_weights)

def eval(y,y_pred,n):
	count = 0
	TP, FP, TN, FN = 0,0,0,0
	for i in range(n):
		if(y[i]<0.5):
			y[i] = 0
		if(y[i]>=0.5):
			y[i] = 1
		if(y[i]==y_pred[i]):
			count+=1
		if(y[i]==1):
			if(y[i]==y_pred[i]):
				TP += 1
			else:
				FP += 1
		if(y[i]==0):
			if(y[i]==y_pred[i]):
				TN += 1
			else:
				FN += 1
	Acc = count/n
	pur_a = TP/(TP+FP)
	pur_not_a = TN/(TN+FN)
	com_a = TP/(TP+FN)
	com_not_a = TN/(TN+FP)
	f1 = 2*TP/(2*TP+FP+FN)
	fpr = FP/(TN+FN)
	tnr = TN/(TN+FN)
	bAcc = (TP/(TP+FP)+TN/(TN+FN))/2.
	k = 2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
	mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	BinBs = (FP+FN)/(TP+FP+FN+TN)

	print(np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs]))
	ev = pd.DataFrame([np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs])], 
    columns=['Accuracy','AGN_purity','nonAGN_precision','AGN_completness','nonAGN_completness','F1',
    'FPR','TNR','bACC','K','MCC','BinaryBS'])

	print("Accuracy 				[worst: 0; best: 1]:",              Acc)
	print("AGN purity 				[worst: 0; best: 1]:",     pur_a)
	print("nonAGN precision 			[worst: 0; best: 1]:",    pur_not_a)
	print("AGN completness 			[worst: 0; best: 1]:",       com_a)
	print("nonAGN completness 			[worst: 0; best: 1]:",     com_not_a)
	print("F1  					[worst: 0; best: 1]:",		f1)
	#print("AGN_F:",     2*(Tq/(Tq+Fq)*Tq/(Tq+Fs))/(Tq/(Tq+Fq)+Tq/(Tq+Fs)) )
	#print("non_AGN_F:",     2*(Ts/(Ts+Fs)*Ts/(Ts+Fq))/(Ts/(Ts+Fq)+Ts/(Ts+Fs)) )
	print("FPR (false positive rate) 		[worst: 1; best: 0]:",		fpr)
	print("TNR (true negative rate) 		[worst: 0; best: 1]:",		tnr)
	print("bACC (balanced accuracy) 		[worst: 0; best: 1]:", bAcc)
	print("K (Cohen's Kappa) 			[worst:-1; best:+1]:",		k)
	print("MCC (Matthews Correlation Coef) 	[worst:-1; best:+1]:",		mcc)
	print("BinaryBS (Brierscore) 			[worst: 1; best: 0]:", BinBs)

	return ev

def ml_volume(train,label,X_train,y_train,X_test,y_test,
	model,optimizer,loss,num_ep,batch_size,validation_split,
	output_path_predict,path_save_eval,name):
	
	model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
	model.fit(X_train, y_train,
		epochs=num_ep,
		verbose=1,
		batch_size=batch_size,
		validation_split=validation_split
		#sample_weight=
		)
	model.evaluate(X_test, y_test, verbose=1)
	model.summary()

	

	Class = model.predict(train, batch_size)
	#print(Class)
	res = pd.DataFrame(np.array(Class), columns=['y_prob'])
	res['Y'] = np.array(label)
	res.to_csv(f'{path_save_eval}_{name}_prob.csv', index=False)

	'''
	g=open(f"{output_path_predict}/{name}.csv",'w')
	Class.tofile(g,"\n")
	g.close()
	'''
	ev = eval(Class,label,label.shape[0])
	ev.to_csv(f'{path_save_eval}_{name}_evaluate.csv', index=False)

	
	
	return model


def NN(train,label,test_size,validation_split,batch_size,num_ep,optimizer,loss,output_path_predict,output_path_mod,output_path_weight,path_save_eval):
#	X_train, X_test, y_train, y_test = train_test_split(train, label, 
#														test_size = test_size, random_state = 56) #0.4
	#print(X_train, X_test, y_train, y_test)
	
	#from keras.utils import np_utils
	#from keras import metrics,losses
	#Y_train = np_utils.to_categorical(y_train, 3)
	#Y_test = np_utils.to_categorical(y_test, 3)
	#print(Y_train, Y_test)
	#batch_size = 1024
	#num_ep = 15
	features = train.shape[1]
	print(features)
	train = np.array(train)
	kfold = KFold(n_splits=5, shuffle=False)
	index=0
	for train_index, test_index in kfold.split(train):
		X_train = train[train_index]
		y_train = label[train_index]
    
		X_test = train[test_index]
		y_test = label[test_index]
	
		model = DeepDenseNN(features)	
		model1 = ml_volume(train,label,X_train,y_train,X_test,y_test,
		model,optimizer,loss,num_ep,batch_size,validation_split,
		output_path_predict,path_save_eval,f"normal_{index}")
		SaveModel(model1,output_path_mod,output_path_weight,f"normal_{index}")

		model = DeepDarkDenseNN(features)
		model2 = ml_volume(train,label,X_train,y_train,X_test,y_test,
		model,optimizer,loss,num_ep,batch_size,validation_split,
		output_path_predict,path_save_eval,f"dark_{index}")
		SaveModel(model2,output_path_mod,output_path_weight,f"dark_{index}")
		'''
		model = NoDeepDenseNN(features)
		model3 = ml_volume(train,label,X_train,y_train,X_test,y_test,
		model,optimizer,loss,num_ep,batch_size,validation_split,
		output_path_predict,path_save_eval,f"no_{index}")
		SaveModel(model3,output_path_mod,output_path_weight,f"no_{index}")
		'''
		index+=1

	'''
	model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
	####
	'''
	'''
	weights_history = []
	import keras
# A custom callback
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
	class MyCallback(keras.callbacks.Callback):
		def on_batch_end(self, batch, logs):
			weights, _biases = model.get_weights()
			w1, w2, w3 = weights
			weights = [w1[0], w2[0], w3[0]]
			#w1, w2, w3, w4, w5, w6, w7 = weights
			#weights = [w1[0], w2[0], w3[0], w4[0], w5[0], w6[0], w7]
			#print('on_batch_end() model.weights:', weights)
			weights_history.append(weights)
	callback = MyCallback()
	'''
	'''
	####
	model.fit(X_train, y_train,
			epochs=num_ep,
			verbose=1,
			batch_size=batch_size,
			validation_split=validation_split)
			#callbacks=[callback]) #0.25
	model.evaluate(X_test, y_test, verbose=1)
	model.summary()
	'''
	'''
	plt.figure(1, figsize=(6, 3))
	plt.plot(weights_history)
	plt.savefig('1231231.jpg')
	'''
	'''
	Class = model.predict(X_test, batch_size)
	#print(Class)
	Class = np.array(Class)
	
	g=open(output_path_predict,'w')
	Class.tofile(g,"\n")
	g.close()

	ev = eval(Class,y_test,y_test.shape[0])
	ev.to_csv(f'{path_save_eval}/ml_evaluate.csv', index=False)
	'''
	'''
	count = 0
	TP, FP, TN, FN = 0,0,0,0
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
				TP += 1
			else:
				FP += 1
		if(Class[i]==0):
			if(Class[i]==y_test[i]):
				TN += 1
			else:
				FN += 1
	print("Accuracy 				[worst: 0; best: 1]:",              count/y_test.shape[0])
	print("AGN purity 				[worst: 0; best: 1]:",     TP/(TP+FP))
	print("nonAGN precision 			[worst: 0; best: 1]:",    TN/(TN+FN))
	print("AGN completness 			[worst: 0; best: 1]:",       TP/(TP+FN))
	print("nonAGN completness 			[worst: 0; best: 1]:",     TN/(TN+FP))
	print("F1  					[worst: 0; best: 1]:",		2*TP/(2*TP+FP+FN))
	#print("AGN_F:",     2*(Tq/(Tq+Fq)*Tq/(Tq+Fs))/(Tq/(Tq+Fq)+Tq/(Tq+Fs)) )
	#print("non_AGN_F:",     2*(Ts/(Ts+Fs)*Ts/(Ts+Fq))/(Ts/(Ts+Fq)+Ts/(Ts+Fs)) )
	print("FPR (false positive rate) 		[worst: 1; best: 0]:",		FP/(TN+FN))
	print("TNR (true negative rate) 		[worst: 0; best: 1]:",		TN/(TN+FN))
	print("bACC (balanced accuracy) 		[worst: 0; best: 1]:", (TP/(TP+FP)+TN/(TN+FN))/2.)
	print("K (Cohen's Kappa) 			[worst:-1; best:+1]:",		2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN)))
	print("MCC (Matthews Correlation Coef) 	[worst:-1; best:+1]:",		(TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
	print("BinaryBS (Brierscore) 			[worst: 1; best: 0]:", (FP+FN)/(TP+FP+FN+TN))
	'''
	