from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import tensorflow as tf

def init():
	json_file = open('static/model_num.json','r')

	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights("static/model_num.h5")
	print("Loaded model")

	loaded_model.compile( optimizer='adam',loss='categorical_crossentropy')
	graph = tf.get_default_graph()

	return loaded_model, graph
