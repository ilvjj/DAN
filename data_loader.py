import numpy as np
from PIL import Image
import torch.utils.data as data
import os

class ALL_tw(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None, rgb_pseudo=None,
                 IR_pseudo=None):
        # Load training images (path) and labels

        color_img_file = []
        train_color_label = []
        thermal_img_file = []
        train_thermal_label = []
        color_pre_list = []
        for i in os.listdir(
                '../Datasets/ThermalWorld_ReID_train_v3_0/train/TV_FULL'):
            color_pre_list.append(
                '../Datasets/ThermalWorld_ReID_train_v3_0/train/TV_FULL' + '/' + i)
        for i in color_pre_list:
            for j in os.listdir(i):
                color_img_file.append(i + '/' + j)
                train_color_label.append(int(i.split('/')[-1]))
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        thermal_pre_list = []
        for i in os.listdir(
                '../Datasets/ThermalWorld_ReID_train_v3_0/train/IR_8'):
            thermal_pre_list.append(
                '../Datasets/ThermalWorld_ReID_train_v3_0/train/IR_8' + '/' + i)
        for i in thermal_pre_list:
            for j in os.listdir(i):
                thermal_img_file.append(i + '/' + j)
                train_thermal_label.append(int(i.split('/')[-1]))

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)



        data_dir = '../Datasets/RegDB/'

        train_color_noise_list = data_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
        train_thermal_noise_list = data_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

        #color_img_file, train_color_label = load_data(train_color_list)
        #thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        noise_color_img_file, _ = load_data(train_color_noise_list)
        noise_thermal_img_file, _ = load_data(train_thermal_noise_list)


        noise_train_color_image = []
        for i in range(len(noise_color_img_file)):
            img = Image.open(data_dir + noise_color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            noise_train_color_image.append(pix_array)
        noise_train_color_image = np.array(noise_train_color_image)

        noise_train_thermal_image = []
        for i in range(len(noise_thermal_img_file)):
            img = Image.open(data_dir + noise_thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            noise_train_thermal_image.append(pix_array)
        noise_train_thermal_image = np.array(noise_train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        # BGR to RGB
        self.noise_train_color_image = noise_train_color_image
        self.noise_train_thermal_image = noise_train_thermal_image
        self.pseudo_color_label = rgb_pseudo
        self.pseudo_thermal_label = IR_pseudo

        self.transform = transform
        self.cIndexs = colorIndex
        self.tIndexs = thermalIndex

        self.cIndext = colorIndex
        self.tIndext = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndexs[index]], self.train_color_label[self.cIndexs[index]]
        img2, target2 = self.train_thermal_image[self.tIndexs[index]], self.train_thermal_label[self.tIndexs[index]]

        if index < len(self.cIndext):
            img3, target3 = self.noise_train_color_image[self.cIndext[index]], self.pseudo_color_label[self.cIndext[index]]
        else:
            img3, target3 = self.noise_train_color_image[self.cIndext[index%(len(self.cIndext))]], self.pseudo_color_label[self.cIndext[index%(len(self.cIndext))]]
        if index < len(self.tIndext):
            img4, target4 = self.noise_train_thermal_image[self.tIndext[index]], self.pseudo_thermal_label[self.tIndext[index]]
        else:
            img4, target4 = self.noise_train_thermal_image[self.tIndext[index%(len(self.tIndext))]], self.pseudo_thermal_label[self.tIndext[index%(len(self.tIndext))]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)
        img4 = self.transform(img4)

        return img1, img2, target1, target2, img3, img4, target3, target4, index

    def __len__(self):
        return len(self.train_color_label)


class ThermalWorld_target(data.Dataset):
    def __init__(self, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = './Datasets/ThermalWorld_ReID_train_v3_0/'

        color_img_file = []
        train_color_label = []
        thermal_img_file = []
        train_thermal_label = []
        color_pre_list = []
        for i in os.listdir('../Datasets/ThermalWorld_ReID_train_v3_0/test/RGB'):
            color_pre_list.append('../Datasets/ThermalWorld_ReID_train_v3_0/test/RGB'+'/'+i)
        for i in color_pre_list:
            for j in os.listdir(i):
                color_img_file.append(i+'/'+j)
                train_color_label.append(int(i.split('/')[-1]))
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        thermal_pre_list = []
        for i in os.listdir(
                '../Datasets/ThermalWorld_ReID_train_v3_0/test/IR'):
            thermal_pre_list.append(
                '../Datasets/ThermalWorld_ReID_train_v3_0/test/IR'+'/' + i)
        for i in thermal_pre_list:
            for j in os.listdir(i):
                thermal_img_file.append(i +'/'+ j)
                train_thermal_label.append(int(i.split('/')[-1]))

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[index], self.train_color_label[index]
        img2, target2 = self.train_thermal_image[index], self.train_thermal_label[index]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2, index

    def __len__(self):
        return len(self.train_color_label)





class ALL_reg_to_tw(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None, rgb_pseudo=None,
                 IR_pseudo=None):
        # Load training images (path) and labels

        noise_color_img_file = []
        noise_train_color_label = []
        noise_thermal_img_file = []
        noise_train_thermal_label = []
        noise_color_pre_list = []
        for i in os.listdir(
                '/media/cqu/D/XYG/Cross-Modal-Re-ID-baseline/base/Datasets/ThermalWorld_ReID_train_v3_0/test/RGB'):
            noise_color_pre_list.append(
                '/media/cqu/D/XYG/Cross-Modal-Re-ID-baseline/base/Datasets/ThermalWorld_ReID_train_v3_0/test/RGB' + '/' + i)
        for i in noise_color_pre_list:
            for j in os.listdir(i):
                noise_color_img_file.append(i + '/' + j)

        noise_train_color_image = []
        for i in range(len(noise_color_img_file)):
            img = Image.open(noise_color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            noise_train_color_image.append(pix_array)
        noise_train_color_image = np.array(noise_train_color_image)

        noise_thermal_pre_list = []
        for i in os.listdir(
                '/media/cqu/D/XYG/Cross-Modal-Re-ID-baseline/base/Datasets/ThermalWorld_ReID_train_v3_0/test/IR'):
            noise_thermal_pre_list.append(
                '/media/cqu/D/XYG/Cross-Modal-Re-ID-baseline/base/Datasets/ThermalWorld_ReID_train_v3_0/test/IR' + '/' + i)
        for i in noise_thermal_pre_list:
            for j in os.listdir(i):
                noise_thermal_img_file.append(i + '/' + j)

        noise_train_thermal_image = []
        for i in range(len(noise_thermal_img_file)):
            img = Image.open(noise_thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            noise_train_thermal_image.append(pix_array)
        noise_train_thermal_image = np.array(noise_train_thermal_image)


        #data_dir = './Datasets/RegDB/'
        data_dir = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB/'

        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label


        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.noise_train_color_image = noise_train_color_image
        self.noise_train_thermal_image = noise_train_thermal_image
        self.pseudo_color_label = rgb_pseudo
        self.pseudo_thermal_label = IR_pseudo

        self.transform = transform
        self.cIndexs = colorIndex
        self.tIndexs = thermalIndex

        self.cIndext = colorIndex
        self.tIndext = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndexs[index]], self.train_color_label[self.cIndexs[index]]
        img2, target2 = self.train_thermal_image[self.tIndexs[index]], self.train_thermal_label[self.tIndexs[index]]

        if index < len(self.cIndext):
            img3, target3 = self.noise_train_color_image[self.cIndext[index]], self.pseudo_color_label[self.cIndext[index]]
        else:
            img3, target3 = self.noise_train_color_image[self.cIndext[index%(len(self.cIndext))]], self.pseudo_color_label[self.cIndext[index%(len(self.cIndext))]]
        if index < len(self.tIndext):
            img4, target4 = self.noise_train_thermal_image[self.tIndext[index]], self.pseudo_thermal_label[self.tIndext[index]]
        else:
            img4, target4 = self.noise_train_thermal_image[self.tIndext[index%(len(self.tIndext))]], self.pseudo_thermal_label[self.tIndext[index%(len(self.tIndext))]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)
        img4 = self.transform(img4)

        return img1, img2, target1, target2, img3, img4, target3, target4, index

    def __len__(self):
        return len(self.train_color_label)



class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        data_dir = '/media/cqu/D/XYG/Cross-domain/Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2#,index

    def __len__(self):
        return len(self.train_color_label)





class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB/'
        train_color_list   = data_dir + 'idx/test_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/test_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):


        # img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        # img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1, target1 = self.train_color_image[index], self.train_color_label[index]
        img2, target2 = self.train_thermal_image[index], self.train_thermal_label[index]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)


        return img1, img2, target1, target2,index

    def __len__(self):
        return len(self.train_color_label)


class RegDBData_source(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        # img1, target1 = self.train_color_image[index], self.train_color_label[index]
        # img2, target2 = self.train_thermal_image[index], self.train_thermal_label[index]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2#, index

    def __len__(self):
        return len(self.train_color_label)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)



def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label









