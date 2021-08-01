#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
#from keras.layers.normalization import BatchNormalization
import os

from keras.layers import * #Input, Dense, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
import numpy as np

def ImageDeepDenseNN(input_shape):
	inputs = Input(shape=input_shape)
    # Image augmentation block
    #x = data_augmentation(inputs)

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
    layer_1 = Dense(64, activation='linear', kernel_initializer='he_uniform' )(input_img)
    layer_2 = Dropout(.5, input_shape=(features,))(layer_1)
    layer_3 = Dense(32, activation='tanh', kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dropout(.5, input_shape=(features,))(layer_3)
    layer_5 = Dense(16, activation='relu', kernel_initializer='he_uniform' )(layer_4)
    layer_6 = Dropout(.5, input_shape=(features,))(layer_5)
    layer_7 = Dense(8, activation='tanh', kernel_initializer='he_uniform' )(layer_6)
    layer_8 = Dropout(.5, input_shape=(features,))(layer_7)
    layer_9 = Dense(4, activation='elu', kernel_initializer='he_uniform' )(layer_8)
    layer_10 = Dropout(.5, input_shape=(features,))(layer_9)
    Label = Dense(1, activation='sigmoid', kernel_initializer='he_uniform')(layer_10)    
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

def PreProcesing(image_size,batch_size,origin_path,main_name,trash_name):
	image_size = (100, 100)
	batch_size = 32

	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		origin_path,
		validation_split=0.2,
		subset="training",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		origin_path,
		validation_split=0.2,
		subset="validation",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)

	import matplotlib.pyplot as plt

	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].np().astype("uint8"))
			plt.title(int(labels[i]))
			plt.axis("off")

	train_ds = train_ds.prefetch(buffer_size=32)
	val_ds = val_ds.prefetch(buffer_size=32)
	return train_ds, val_ds

def ImageNN(image_size,train_ds,val_ds,epochs,path_model,path_weights):
	model = ImageDeepDenseNN(image_size)
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
	model.fit(train_ds, epochs=epochs, validation_data=val_ds)
	model.summary()
	SaveModel(model,path_model,path_weights)

def TestNN(path_model,path_weights,test_path,image_size,batch_size):
	model = LoadModel(path_model,path_weights)
	test_sample = tf.keras.preprocessing.image_dataset_from_directory(
		test_path,
		image_size=image_size,
		batch_size=batch_size
	)
	return model.predict(test_sample, batch_size)
	

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