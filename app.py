import numpy as np
import warnings
import json
warnings.simplefilter(action='ignore', category=FutureWarning)

from flask import Flask
from flask import request
from datetime import date
import datetime
import tensorflow as tf
import ast
import base64
from io import BytesIO
from PIL import Image

tf.logging.set_verbosity(tf.logging.ERROR)

import plastic_dict
from imageai.Detection import ObjectDetection
# import object_detection

user_objects = [["fork", 6, datetime.datetime(2020, 1, 12)],
				["backpack", 500, datetime.datetime(2020, 2, 12)],
				["cup", 75, datetime.datetime(2020, 3, 15)],
				["fork", 6, datetime.datetime(2020, 4, 1)],
				["fork", 6, datetime.datetime(2020, 5, 15)],
				["cellphone", 87, datetime.datetime(2020, 7, 18)],
				]

def no_input():
	'''
	Function that returns when no image is given
	'''

	return "N/A - No Image Given"

app = Flask(__name__)


def str_to_lol(string_lol):
	strs = string_lol.replace('[','').split('],')
	lists = [list(map(int, s.replace(']','').split(','))) for s in strs]
	return lists

def get_plastic_amounts(detection_obj):
	'''
	Function to get the amount of plastic for elements in a list
	'''

	return [object_plastic_map[x["name"]] for x in detection_obj]

@app.route('/predict', methods=['POST', 'GET'])
def predict():

	if request.method == 'POST':
		from keras import backend as K
		K.clear_session()
		print("POST")

		# loading model here in background
		detector = ObjectDetection()

		# use fat yolo since tiny yolo sux
		detector.setModelTypeAsYOLOv3()
		path = "./models/yolo.h5"

		detector.setModelPath(path)
		detector.loadModel()

		# creating list of custom objects
		custom = detector.CustomObjects(backpack=True, umbrella=True, cup=True)

		object_plastic_map = plastic_dict.plastic_dict

		# req_data = request.get_json()
		# image = req_data["image"]

		print(list(request.form.keys()))

		str_image = request.form["image"]
		print(str_image[:20])
		str_image = base64.b64decode(str_image)
		print(str_image[:20])

		# str_image = str_image.decode('utf-8')

		# str_image = base64_bytes.decode("base64")
		# print(str_image)

		# converting bytes information into string into np array
		# str_image = image.decode("utf-8")
		# print(str_image)



		image = Image.open(BytesIO(str_image)).convert("RGB")

		# image = ast.literal_eval(str_image)
		# print(image)
		im_array = np.array(image,dtype=np.uint8)
		# print(np.shape(image))
		print(np.shape(im_array))
		detection = detector.detectCustomObjectsFromImage(custom_objects=custom,
												  input_type="array", input_image=im_array,
												  output_type="array",
												  minimum_percentage_probability=70)

		K.clear_session()

		to_return = ""
		for eachItem in detection[1]:
			name = eachItem["name"]
			print(name + " : ", eachItem["percentage_probability"])
			to_return += name
			# to_return += ":"
			# to_return += str(object_plastic_map[name])
			to_return += str(",")

		if to_return == "":
			print("nothing detected")

		print(to_return)

		return to_return
	else:
		return no_input()

# function to add to list
@app.route('/add', methods=['POST', 'GET'])
def add():

	if request.method == 'POST':

		req_data = request.get_json()
		obj = req_data["object"]
		plastic_amt = req_data["plastic"]
		dt = req_data["date"]
		dt = datetime.datetime(dt[0], dt[1], dt[2])
		user_objects.append([obj, plastic_amt, dt])
	return "Added Succesfully!"

# function to get list
@app.route('/list', methods=['GET'])
def get_list():
	if request.method == 'GET':
		return str(user_objects)
	else:
		return "Rip Big Fail"

@app.route('/')
def main():
	return 'App Running!'

if __name__ == '__main__':
	app.run(debug=True)
