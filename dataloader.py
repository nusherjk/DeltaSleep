import os
import numpy as np
from torch.utils import  data
from torchvision import transforms
from torch.utils.data import DataLoader
from augmentation import *
from lxml import etree



import warnings
warnings.filterwarnings("ignore")


####################################### CUSTOM DATASET FOR DATALOADER ##################################################

#label_map = {'Car': 1, 'Bus': 2, 'Truck': 3, 'CNG': 4, 'Pickup':5}
#batch_size = 16

def iterate_through_imgs(path):
    #path is the img path for  imgs folder
    img_ids = []
    for imgs in os.listdir(path):
        img_name = os.path.splitext(imgs)[0]
        img_ids.append(img_name)
    return img_ids

def load_bboxs(path_xml):
    currentdata = []
    bboxs = []
    label_map = np.array(['Car', 'Bus', 'Truck', 'CNG', 'Pickup'])
    # path_xml does not contain the .xml extention
    extention = '.xml'
    filename = path_xml + extention

    tre = etree.parse(filename)
    root = tre.getroot()


    data = root.find('size')
    width = data.find('width').text
    height = data.find('height').text
    depth = data.find('depth').text


    for objects in root.iter('object'):
        label = objects.find('name').text.lower().strip()
        #print(label)
        bndbox = objects.find('bndbox')
        label_m = {'car': 0, 'bus': 1, 'truck': 2, 'cng': 3, 'pickup': 4}
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        #label_id = np.argwhere(label_map== label)[0][0]

        lid = label_m[label.lower()]
        currentdata = np.array([lid,xmin,ymin,xmax,ymax] )
        bboxs.append(currentdata)

        #bbox.append([xmin,ymin,xmax,ymax])

    #read each class and bbox coordinates

    return np.array(bboxs)


class MainDataset(data.Dataset):


    def __init__(self, path_to_imgs, path_to_xml, transform=None, max_truth=30):

        self.path_img = path_to_imgs
        self.path_xml = path_to_xml
        self.img_ids = iterate_through_imgs(path_to_imgs)
        self.transform = transform
        self.max_truth = max_truth


    def __getitem__(self, item):
        img_id = self.img_ids[item]
        bboxs = load_bboxs(os.path.join(self.path_xml, img_id))
        imgname = img_id + '.jpg'
        img = cv2.imread(os.path.join(self.path_img, imgname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.asarray(img)

        dictforarray = {"image": img_array, "bboxes": bboxs}

        if self.transform:
            dictforarray = self.transform(dictforarray)
        bboxes = dictforarray['bboxes'].numpy()
        n_truth = len(bboxes)

        dictforarray['bboxes'] = torch.from_numpy(bboxes)
        dictforarray['n_true'] = n_truth
        return dictforarray

    def __len__(self):
        counter =0
        for name in os.listdir(self.path_img):
            counter+=1
        return counter




def getdatasets(path_dataset, batch_size, image_size=416 ):


    transform_fn = transforms.Compose([

                                       Rescale((image_size, image_size)),
                                       TransformBoxCoords(),
                                       Normalize(),
                                       EliminateSmallBoxes(0.025),
                                       ToTensor()])
    train_location = path_dataset + '/Train/'
    test_location = path_dataset +  "/Test/"
    val_location = path_dataset + "/Validation/"


    train_img = train_location + '/' + "i"
    train_xml = train_location + '/' + "a"

    test_img = test_location + '/' + "i"
    test_xml = test_location + '/' + 'a'

    val_img = val_location + '/' + 'i'
    val_xml = val_location +'/' + 'a'

    train = MainDataset(train_img, train_xml, transform=transform_fn)

    test = MainDataset(test_img, test_xml, transform=transform_fn)
    val = MainDataset(val_img, val_xml,transform=transform_fn)


    train_loader = DataLoader(train,batch_size=batch_size,shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test,batch_size=batch_size,shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val,batch_size=batch_size,shuffle=True, num_workers=4, drop_last=True)

    return train_loader, test_loader, val_loader
