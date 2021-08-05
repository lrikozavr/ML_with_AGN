# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Dropout
from keras.layers import experimental, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, add
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from settings import *

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
		color_mode="grayscale",
		image_size=image_size,
		batch_size=batch_size
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		origin_path,
		validation_split=0.2,
		subset="validation",
		seed=1337,
		color_mode="grayscale",
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
	train_ds = train_ds.prefetch(buffer_size=32)
	val_ds = val_ds.prefetch(buffer_size=32)

	return train_ds, val_ds

def ImageNN(image_size,batch_size,train_ds,val_ds,epochs,path_model,path_weights):
	model = ImageDeepDenseNN(input_shape=image_size + (1,))
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

def Image_ML(origin_path, image_size, batch_size, epochs, path_model, path_weights):
	train_ds, val_ds = PreProcesing(image_size,batch_size,origin_path)
	ImageNN(image_size,batch_size,train_ds,val_ds,epochs,path_model,path_weights)
			
def Start_IMG(MODEL_PATH,SAVE_PATH):
	image_size = (IMG_SIZE, IMG_SIZE)
	batch_size = BATCH_SIZE
	epochs = EPOCHS
	format_ = "jpg"
	main_path = f"{SAVE_PATH}/{format_}"
	model_path = MODEL_PATH

	for band in os.listdir(main_path):
		band_path = f"{main_path}/{band}"
		path_model = f"{model_path}/model_img_{band}"
		path_weights = f"{model_path}/weights_img_{band}"
		_name = os.listdir(origin_path)
		PreDelete(band_path,_name[0],_name[1])
		Image_ML(band_path, image_size, batch_size, epochs, path_model, path_weights)
		for name in [ "AGN", "GALAXY" ]:
			test_path = f"{SAVE_PATH}/{name}/{format_}/{band}"
			TestNN(test_path,image_size,batch_size,path_model,path_weights)
