import requests
import numpy as np
from PIL import Image

def test_obj_detection():
	im = Image.open("test_images/103656324_190160012323275_1284606866924854408_n.png")
	image = np.array(im)
	image = image.tolist()
	r = requests.post('http://127.0.0.1:5000/predict', json={'image':image})
	print(r.text)

def test_add_item():
	list_before = requests.get('http://127.0.0.1:5000/list')
	print(list_before.text)
	r = requests.post('http://127.0.0.1:5000/add', json={'object': "fork", "plastic": 7, "date": (2020, 10, 3)})
	list_after = requests.get('http://127.0.0.1:5000/list')
	print(list_after.text)

if __name__ == "__main__":
	# test_add_item()
	test_obj_detection()