import requests
import numpy as np
from PIL import Image

im = Image.open("test_images/103656324_190160012323275_1284606866924854408_n.png")
image = np.array(im)
image = image.tolist()
r = requests.post('http://127.0.0.1:5000/predict', json={'image':image})
print(r)
