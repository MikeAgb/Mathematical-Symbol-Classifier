
from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
from skimage import img_as_bool, io, color, morphology
import numpy as np 
import keras.models
import re
import sys
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
import cv2
import base64

#dictionary of labels
labels = {0: '!', 1: '(', 2: ')', 3: '+', 4: '-', 5: '0', 6: '1', 7: '2', 8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9', 15: '=', 16: 'A', 17: 'C', 18: 'Delta', 19: 'G', 20: 'H', 21: 'M', 22: 'N', 23: 'R', 24: 'S', 25: 'T', 26: 'X', 27: '[', 28: ']', 29: 'alpha', 30: 'b', 31: 'beta', 32: 'cos', 33: 'd', 34: 'div', 35: 'e', 36: 'f', 37: 'forward_slash', 38: 'gamma', 39: 'geq', 40: 'gt', 41: 'i', 42: 'infty', 43: 'int', 44: 'j', 45: 'k', 46: 'l', 47: 'lambda', 48: 'ldots', 49: 'leq', 50: 'lim', 51: 'log', 52: 'lt', 53: 'mu', 54: 'neq', 55: 'p', 56: 'phi', 57: 'pi', 58: 'pm', 59: 'q', 60: 'rightarrow', 61: 'sigma', 62: 'sin', 63: 'sqrt', 64: 'sum', 65: 'tan', 66: 'theta', 67: 'times', 68: 'u', 69: 'v', 70: 'w', 71: 'y', 72: 'z', 73: '{', 74: '}'}


sys.path.append(os.path.abspath('./model'))  #model folder

#load import.py
from load import *

#initialise app
app = Flask(__name__)
global model, graph
model, graph = init()

#landing page
@app.route('/')
def index():
	return render_template('index.html')

def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',str(imgData1)).group(1)
	with open('out.png','wb') as output:
		output.write(base64.b64decode(imgstr))

#redicred when they hit predict
@app.route('/predict/', methods = ['GET', 'POST'])
def predict():

	imgData = request.get_data()
  
  #call convert image on image
	convertImage(imgData)

  #save the image as out.png abd apply transformations
	im = cv2.imread('out.png')
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

  #threshold the image, and find the contours
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
	threshed = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, rect_kernel)
	ctrs, hier = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  #sort the contours in linear order
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

	output=""

	for i, ctr in enumerate(sorted_ctrs):
		x, y, w, h = cv2.boundingRect(ctr)
		roi = im[y:y+h, x:x+w]
		area = w*h
     
    #only predict if the area is large enough, if smaller than 200, it is probably a mistake
		if 200 < area:
			cv2.imwrite(str(i) +  '.jpg', roi)
			test_image = image.load_img(str(i) + '.jpg',color_mode = "grayscale",target_size= (45,45))
			test_image = image.img_to_array(test_image)
      #expand dimension to be able to feed the image into the model
			test_image = np.expand_dims(test_image, axis = 0)

			with graph.as_default():
				out = model.predict(test_image)
				print(out)
				answer=np.argmax(out, axis=1)
				output+=labels[answer[0]] + " "
        #delete the saved images
				os.remove(str(i) +  '.jpg')


	return output

  #apply transformations to the individual images  for predictions
	x = imresize(x, (45, 45))
	x = x.reshape(1, 45, 45, 1)

	with graph.as_default():

		out = model.predict(x)
		print(out)
		print(np.argmax(out, axis=1))

		response = np.argmax(out, axis=1)
		return labels[response[0]]
        	 


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port = port)

