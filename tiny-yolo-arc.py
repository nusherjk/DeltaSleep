import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F
import cv2


from dataloader import getdatasets
from Yololoss import Yolov2Loss
from utils2 import  get_map
from optimizer import get_optimizer

def get_meta():
    batch_size = 16
    meta = {}
    meta['anchors'] = 5
    meta['classes'] = 5
    meta['batch_size'] = batch_size
    meta['threshold'] = 0.6
    meta['anchor_bias'] = np.array([1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52])
    meta['scale_no_obj'] = 1
    meta['scale_coords'] = 1
    meta['scale_class'] = 1
    meta['scale_obj'] = 5
    meta['iteration'] = 0
    meta['train_samples'] = 45831
    meta['iterations_per_epoch'] = meta['train_samples'] / batch_size
    return meta

#tiny yolo v2 arcitecture by our design
class TinyYolo(nn.Module):
    def __init__(self):
        super(TinyYolo, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.layer5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.layer6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.layer7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(1024)
        self.layer8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer9 = nn.Conv2d(in_channels=1024, out_channels=50, kernel_size=1, stride=1, padding=1, bias=False)
        #self.batchnorm8 = nn.BatchNorm2d(125)
        self.pool = nn.MaxPool2d(2,2)
        self.lReLU = nn.LeakyReLU()



    def forward(self, x):

        out = self.lReLU(self.batchnorm1(self.layer1(x)))
        out = self.pool(out)
        out = self.lReLU(self.batchnorm2(self.layer2(out)))
        out = self.pool(out)
        out = self.lReLU(self.batchnorm3(self.layer3(out)))
        out = self.pool(out)
        out = self.lReLU(self.batchnorm4(self.layer4(out)))
        out = self.pool(out)
        out = self.lReLU(self.batchnorm5(self.layer5(out)))
        out = self.pool(out)
        out = self.lReLU(self.batchnorm6(self.layer6(out)))
        out = self.pool(out)
        out = self.lReLU(self.batchnorm7(self.layer7(out)))
        out = self.lReLU(self.batchnorm7(self.layer8(out)))
        out = self.layer9(out)

        return out

#classic Yolov2 29 layer design...
class ObjectDetectbyYolov2(nn.Module):
    def __init__(self):
        super(ObjectDetectbyYolov2,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm13 = nn.BatchNorm2d(512)

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm14 = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm16 = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm18 = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm19 = nn.BatchNorm2d(1024)
        self.conv20 = nn.Conv2d(in_channels=1024, out_channels=3072, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm20 = nn.BatchNorm2d(3072)

        self.conv21 = nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm21 = nn.BatchNorm2d(1024)

        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=50, kernel_size=1, stride=1, padding=0)

    def reorg_layer(self, x):
        stride = 2
        batch_size, channels, height, width = x.size()
        new_ht = height / stride
        new_wd = width / stride
        new_channels = channels * stride * stride

        passthrough = x.permute(0, 2, 3, 1)
        passthrough = passthrough.contiguous().view(-1, new_ht, stride, new_wd, stride, channels)
        passthrough = passthrough.permute(0, 1, 3, 2, 4, 5)
        passthrough = passthrough.contiguous().view(-1, new_ht, new_wd, new_channels)
        passthrough = passthrough.permute(0, 3, 1, 2)
        return passthrough

    def forward(self, x):
        out = F.max_pool2d(F.leaky_relu(self.batchnorm1(self.conv1(x)), negative_slope=0.1), 2, stride=2)
        out = F.max_pool2d(F.leaky_relu(self.batchnorm2(self.conv2(out)), negative_slope=0.1), 2, stride=2)

        out = F.leaky_relu(self.batchnorm3(self.conv3(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm4(self.conv4(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm5(self.conv5(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm6(self.conv6(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm7(self.conv7(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm8(self.conv8(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm9(self.conv9(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm10(self.conv10(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm11(self.conv11(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm12(self.conv12(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm13(self.conv13(out)), negative_slope=0.1)
        #passthrough = self.reorg_layer(out)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm14(self.conv14(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm15(self.conv15(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm16(self.conv16(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm17(self.conv17(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm18(self.conv18(out)), negative_slope=0.1)

        out = F.leaky_relu(self.batchnorm19(self.conv19(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm20(self.conv20(out)), negative_slope=0.1)

        #out = torch.cat([passthrough, out], 1)
        out = F.leaky_relu(self.batchnorm21(self.conv21(out)), negative_slope=0.1)
        out = self.conv22(out)

        return out









def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)


    # Convert to Variable
    return img_



def train(model, train_data, opt=None, iou_thresh= None):
    train_images = Variable(train_data["image"], requires_grad=True).float()
    train_labels = Variable(train_data["bboxes"], requires_grad=False).float()
    train_n_true = train_data["n_true"]
    opt.zero_grad()
    train_output = model(train_images)


    loss = Yolov2Loss(train_output, train_labels, train_n_true.numpy(), get_meta())
    loss.backward()

    opt.step()
    #nms = get_nms_boxes(train_output, 0.3, 0.2, get_meta())
    #print(nms)
    train_map = get_map(train_output, train_labels, train_n_true, iou_thresh, get_meta())
    #train_map = 0
    return loss, train_map
    #return loss,0

def input_stream():
    model = TinyYolo()
    train_loader, test_loader, val_loader = getdatasets('./data/', batch_size=16)


    for i, train_data in enumerate(train_loader):
        opt = get_optimizer(model, [0.00001])
        out,train_map = train(model, train_data, opt=opt, iou_thresh=0.1 )
        print( out)
        print(train_map)
        print()
        print()


def training(epoch):
    for i in len(epoch):
        print("epochs: " + i)
        loss, train_map = input_stream()
        print("Loss: " + loss)
        print('Train MAP: ' + train_map)


if __name__== '__main__':
    '''
    #model = ObjectDetectbyYolov2()
    model = TinyYolo()
    out = model(get_test_input())
    print(out.shape)
    '''
    input_stream()





