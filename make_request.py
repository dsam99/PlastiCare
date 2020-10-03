import requests
import numpy as np
from PIL import Image

def pred_request(image_list):
	r = requests.post('http://127.0.0.1:5000/predict', 
					  json={'image':image_list})
	obj_list = r.text[:-2].split(",")
	return [x.split(":") for x in obj_list] 

def request_add(info):

	r = requests.post('http://127.0.0.1:5000/add', 
					  json={'object': info[0], "plastic": info[1], "date": info[2]})

def request_get():
	r = requests.get('http://127.0.0.1:5000/list')
	return r.text

if __name__ == "__main__":
	im = Image.open("test_images/test_backpack.jpg")
	image = np.array(im)
	image = image.tolist()

	print(pred_request(image))
	print(request_get())