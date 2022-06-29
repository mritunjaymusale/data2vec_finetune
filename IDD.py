import glob
import os
import torch.utils.data as data
from ImageUtils import ImagePreprocessing
from torch.utils.data import DataLoader



class IDD(data.Dataset):
    part1_nested_image_folder_path: list
    part1_nested_mask_folder_path: list
    part2_nested_image_folder_path: list
    part2_nested_mask_folder_path: list

    def __init__(self, base_folder_path: str, resize_size: int, transform=None,test=False):
        super(IDD, self).__init__()
        self.base_folder_path = base_folder_path
        self.is_test = test
        self.transform = transform
        self.resize_size = resize_size

        # for part1 of the dataset
        self.part1_image_files_list = glob.glob(os.path.join(
            self.base_folder_path, *self.part1_nested_image_folder_path))

        self.part1_mask_files_list = glob.glob(os.path.join(
            self.base_folder_path, *self.part1_nested_mask_folder_path))

        # for part2 of the dataset
        self.part2_image_files_list = glob.glob(os.path.join(
            self.base_folder_path, *self.part2_nested_image_folder_path))

        self.part2_mask_files_list = glob.glob(os.path.join(
            self.base_folder_path, *self.part2_nested_mask_folder_path))


        # merge the two lists
        self.image_files_list = self.part1_image_files_list + self.part2_image_files_list
        self.mask_files_list = self.part1_mask_files_list + self.part2_mask_files_list

        # need to do this or the fetching of files is random
        self.image_files_list.sort()
        self.mask_files_list.sort()

        if len(self.image_files_list) != len(self.mask_files_list) and  self.is_test ==False:
            raise Exception('The number of images and masks do not match {} vs {}'.format(
                len(self.image_files_list), len(self.mask_files_list)))

    def __getitem__(self, index):
        img_path = self.image_files_list[index]

        image = ImagePreprocessing.read_image(img_path)
        image = ImagePreprocessing.resizeImg(image, self.resize_size)
        image = ImagePreprocessing.convertImgToTensor(image)

        if self.transform:
            image = self.transform(image)

         
        if self.is_test:
            return image,img_path
        else:
            mask_path = self.mask_files_list[index]
            mask = ImagePreprocessing.read_image(mask_path, is_greyscale=True)
            mask = ImagePreprocessing.resizeImg(mask, self.resize_size)
            mask = ImagePreprocessing.convertImgToTensor(mask, is_greyscale=True)
            return image, img_path,mask,mask_path
        

    def __len__(self):
        if self.is_test:
            return len(self.image_files_list)
        else:
            return len(self.mask_files_list)


#! DO NOT CHANGE JPG TO PNG HERE DATASET CREATOR MADE THE SECOND DATASET IMAGES WITH JPG INSTEAD OF PNG
class IDD_Train(IDD):
    part1_nested_image_folder_path = ['IDD_Segmentation',
                                'leftImg8bit', 'train', '**', '*.png']
    part1_nested_mask_folder_path = ['IDD_Segmentation',
                               'gtFine', 'train', '**', '*.png']

    part2_nested_image_folder_path = ['idd20kII',
                                'leftImg8bit', 'train', '**', '*.jpg']
    part2_nested_mask_folder_path = ['idd20kII', 'gtFine', 'train', '**', '*.png']

class IDD_Val(IDD):
    part1_nested_image_folder_path = ['IDD_Segmentation',
                                'leftImg8bit', 'val', '**', '*.png']
    part1_nested_mask_folder_path = ['IDD_Segmentation',
                               'gtFine', 'val', '**', '*.png']

    part2_nested_image_folder_path = ['idd20kII',
                                'leftImg8bit', 'val', '**', '*.jpg']
    part2_nested_mask_folder_path = ['idd20kII', 'gtFine', 'val', '**', '*.png']

class IDD_Test(IDD):
    part1_nested_image_folder_path = ['IDD_Segmentation',
                                'leftImg8bit', 'test', '**', '*.png']
    part1_nested_mask_folder_path = ['IDD_Segmentation',
                               'gtFine', 'test', '**', '*.png']

    part2_nested_image_folder_path = ['idd20kII',
                                'leftImg8bit', 'test', '**', '*.jpg']
    part2_nested_mask_folder_path = ['idd20kII', 'gtFine', 'test', '**', '*.png']




class IDDDataLoaders(object):

    @staticmethod
    def get_train_dataloader(hparams):
        train_dataset = IDD_Train(
            hparams['data_data_src'], resize_size=hparams['data_image_resize_size'])

        train_loader = DataLoader(
            train_dataset, batch_size=hparams['data_batch_size'], shuffle=hparams['data_shuffle'], num_workers=hparams['data_num_workers'], pin_memory=hparams['data_pin_memory'], drop_last=True)

        return train_loader

    @staticmethod
    def get_val_dataloader(hparams):
        val_dataset = IDD_Val(
            hparams['data_data_src'], resize_size=hparams['data_image_resize_size'])

        val_loader = DataLoader(
            val_dataset, batch_size=hparams['data_batch_size'], shuffle=hparams['data_shuffle'], num_workers=hparams['data_num_workers'], pin_memory=hparams['data_pin_memory'], drop_last=True)

        return val_loader

    @staticmethod
    def get_test_dataloader(hparams):
        test_dataset = IDD_Test(
            hparams['data_data_src'], resize_size=hparams['data_image_resize_size'], test=True)

        test_loader = DataLoader(
            test_dataset, batch_size=hparams['data_batch_size'], shuffle=hparams['data_shuffle'], num_workers=hparams['data_num_workers'], pin_memory=hparams['data_pin_memory'], drop_last=True)

        return test_loader


if __name__ == '__main__':
    root_src = './'
    data_src = root_src + 'IDD'

    
    dataset = IDD_Train(
        data_src, resize_size=224)
    print(len(dataset))
    
    dataset = IDD_Val(
        data_src, resize_size=224)
    print(len(dataset))

    dataset = IDD_Test(
        data_src, resize_size=224,test=True)
    print(len(dataset))
    