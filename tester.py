from __future__ import division

from tiny-yolo-arc import *
from Darknet import *
import cv2
#from datloader import MainDataset
import  torch

from utils import *
from dataloader import *


from torch.autograd import Variable
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#meta info stored in net of darknet
meta = {'channels': '3',
        'hue': '.1',
        'steps': '400000,450000',
        'subdivisions': '1',
        'width': '416',
        'height': '416',
        'policy': 'steps',
        'burn_in': '1000',
        'saturation': '1.5',
        'exposure': '1.5',
        'decay': '0.0005',
        'angle': '0',
        'batch': '1',
        'learning_rate': '0.001',
        'max_batches': '500200',
        'type': 'net',
        'momentum': '0.9',
        'scales': '.1,.1'}


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
'''
model = Yolov3("yolov3-custom.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())



print(pred)

'''

#train_loader, test_loader, val_loader = getdatasets('./data/', batch_size=meta['batch'])

def train(model, train_data, opt=None, iou_thresh= None):
    train_images = Variable(train_data["image"], requires_grad=True).float()
    train_labels = Variable(train_data["bboxes"], requires_grad=False).float()
    train_n_true = train_data["n_true"]
    #opt.zero_grad()
    train_output = model(train_images, False)
    #loss = Yolov2Loss(train_output, train_labels, train_n_true.numpy())
    #loss.backward()
    #opt.step()
    #train_map = get_map(train_output, train_labels, train_n_true, iou_thresh)
    #return loss, train_map
    return train_output

def input_stream():
    model = Yolov3("yolov3-custom.cfg")
    train_loader, test_loader, val_loader = getdatasets('./data/', batch_size=int(meta['batch']))

    for i, train_data in enumerate(train_loader):
        output = train(model, train_data)
        print(output.shape)



input_stream()

