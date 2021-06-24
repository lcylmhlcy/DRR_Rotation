from PIL import Image
import numpy as np

img_path = '/home/computer/LCY/yiliao/bone_data/axz/axz_5.JPG'
img = Image.open(img_path).convert('L') 
# print(img.shape)
img = np.expand_dims(np.asarray(img), 0)
print(img.shape)