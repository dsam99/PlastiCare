import numpy as np

from flask import Flask
from flask import request

from imageai.Detection import ObjectDetection


app = Flask(__name__)

# loading model here in background
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()

path = "./models/yolo-tiny.h5"
detector.setModelPath(path)
detector.loadModel()

object_plastic_map = load_object_plastic_map()

# creating list of custom objects
custom = detector.CustomObjects(backpack=True, umbrella=True, handbag=True, tie=True, toothbrush=True, cup=True, 
								fork=True, knife=True, spoon=True, suitcase=True, tennis_racket=True, chair=True, 
								remote=True, mouse=True, keyboard=True, cell_phone=True, scissors=True)

def load_object_plastic_map():
	'''
	Function to load a dictionary that is the amount of plastic in a given item
	'''

	return {}

def no_input():
	'''
	Function that returns when no image is given
	'''

	return "N/A - No Image Given"

def get_plastic_amounts(detection_obj):
	'''
	Function to get the amount of plastic for elements in a list
	'''

	return [object_plastic_map[x["name"]] for x in detection_obj]

@app.route('/predict', methods=['POST', 'GET'])
def predict():
	if request.method == 'POST':
		image = request.form["image"]
		image = OD.preprocess(image)

		# percentage threshold of 70%
		detection = detector.detectCustomObjectsFromImage(custom_objects=custom, 
														  input_type="array", input_image=image, 
														  minimum_percentage_probability=70)
		
		for eachItem in detection:
    		print(eachItem["name"] , " : ", eachItem["percentage_probability"])
    		# print(detection)
		return detection
	else:
		return no_input()

@app.route('/')
def main():
    return 'App Running!'

if __name__ == '__main__':
	app.run(debug=True)


