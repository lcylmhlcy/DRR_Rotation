from torch.utils.data import Dataset
from torch import nn
import numpy as np
from PIL import Image

def get_imgs(img_txt='../bone_process/bone.txt'):
    fh = open(img_txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        imgs.append(line)
    fh.close()
    return imgs

class BoneDataSet(nn.Module):
    def __init__(self):
        super(BoneDataSet, self).__init__()
        self.img_dir = '../bone_data/'
        self.imgs = get_imgs()
        
    def __getitem__(self, index):
        img_name = self.imgs[index]
        img, ext = img_name.split('.')
        person, angle = img.split('_')
        target_name = person + '_label.' + ext
        img_path = self.img_dir + person + '/' + img_name
        target_path = self.img_dir + person + '/' + target_name
        # print(img_path)
        # print(target_path)

        img = Image.open(img_path).convert('L').resize((224, 224),Image.ANTIALIAS)
        target = Image.open(target_path).convert('L').resize((224, 224),Image.ANTIALIAS)
        img = np.asarray(img, dtype=np.float32).copy()
        target = np.asarray(target, dtype=np.float32).copy() / 255

        if int(angle) == 5:
            angle = 0
        elif int(angle) == 10:
            angle = 1
        elif int(angle) == 15:
            angle = 2
        elif int(angle) == 25:
            angle = 3
        elif int(angle) == 30:
            angle = 4
        elif int(angle) == 35:
            angle = 5
        elif int(angle) == 40:
            angle = 6
        angle = np.asarray(angle)

        return img, angle, target
 
    def __len__(self):
        return len(self.imgs)