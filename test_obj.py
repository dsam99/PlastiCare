import numpy as np
from PIL import Image
from imageai.Detection import ObjectDetection

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# loading image
im = Image.open("test_images/103656324_190160012323275_1284606866924854408_n.png")
# im = Image.open("test_images/cups.jpg")
image = np.array(im)

# loading model here in background
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()

path = "./models/yolo.h5"
detector.setModelPath(path)
detector.loadModel()

# creating list of custom objects
custom = detector.CustomObjects(backpack=True, umbrella=True, handbag=True, tie=True, toothbrush=True, cup=True, 
								fork=True, knife=True, spoon=True, suitcase=True, tennis_racket=True, chair=True, 
								remote=True, mouse=True, keyboard=True, cell_phone=True, scissors=True, car=True)


detection = detector.detectCustomObjectsFromImage(custom_objects=custom, 
												  input_type="array", input_image=image, 
												  output_type="array",
												  minimum_percentage_probability=70)

# detection = detector.detectObjectsFromImage(input_type="array", input_image=image, 
												  # output_type="array",
												  # minimum_percentage_probability=70)

# for eachItem in detection:
	# print(eachItem["name"] , " : ", eachItem["percentage_probability"])
print(detection)