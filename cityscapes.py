from torch.utils.data import DataLoader
import glob
import os
import torch.utils.data as data
from ImageUtils import ImagePreprocessing



# altered from IDD.py
class Cityscapes(data.Dataset):
    nested_image_folder_path: list
    nested_mask_folder_path: list

    def __init__(self, base_folder_path: str, resize_size: int, transform=None, test=False):
        super(Cityscapes, self).__init__()
        self.base_folder_path = base_folder_path
        self.is_test = test
        self.transform = transform
        self.resize_size = resize_size

        # for part1 of the dataset
        self.image_files_list = glob.glob(os.path.join(
            self.base_folder_path, *self.nested_image_folder_path))

        self.mask_files_list = glob.glob(os.path.join(
            self.base_folder_path, *self.nested_mask_folder_path))

        # need to do this or the fetching of files is random
        self.image_files_list.sort()
        self.mask_files_list.sort()

        if len(self.image_files_list) != len(self.mask_files_list) and self.is_test == False:
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
            return image, img_path
        else:
            mask_path = self.mask_files_list[index]
            mask = ImagePreprocessing.read_image(mask_path, is_greyscale=True)
            mask = ImagePreprocessing.resizeImg(mask, self.resize_size)
            mask = ImagePreprocessing.convertImgToTensor(
                mask, is_greyscale=True)
            return image, img_path, mask, mask_path

    def __len__(self):
        if self.is_test:
            return len(self.image_files_list)
        else:
            return len(self.mask_files_list)


class Cityscapes_Train(Cityscapes):
    nested_image_folder_path = [
        'leftImg8bit', 'train', '**', '*.png']
    nested_mask_folder_path = [
        'gtFine', 'train', '**', '*_labelTrainIds.png']


class Cityscapes_Val(Cityscapes):
    nested_image_folder_path = [
        'leftImg8bit', 'val', '**', '*.png']
    nested_mask_folder_path = [
        'gtFine', 'val', '**', '*_labelTrainIds.png']


class Cityscapes_Test(Cityscapes):
    nested_image_folder_path = [
        'leftImg8bit', 'test', '**', '*.png']
    nested_mask_folder_path = [
        'gtFine', 'test', '**', '*_labelTrainIds.png']


class CityscapesDataLoaders(object):

    @staticmethod
    def get_train_dataloader(hparams):
        train_dataset = Cityscapes_Train(
            hparams['data_data_src'], resize_size=hparams['data_image_resize_size'])

        train_loader = DataLoader(
            train_dataset, batch_size=hparams['data_batch_size'], shuffle=hparams['data_shuffle'], num_workers=hparams['data_num_workers'], pin_memory=hparams['data_pin_memory'], drop_last=True)

        return train_loader

    @staticmethod
    def get_val_dataloader(hparams):
        val_dataset = Cityscapes_Val(
            hparams['data_data_src'], resize_size=hparams['data_image_resize_size'])

        val_loader = DataLoader(
            val_dataset, batch_size=hparams['data_batch_size'], shuffle=hparams['data_shuffle'], num_workers=hparams['data_num_workers'], pin_memory=hparams['data_pin_memory'], drop_last=True)

        return val_loader

    @staticmethod
    def get_test_dataloader(hparams):
        test_dataset = Cityscapes_Test(
            hparams['data_data_src'], resize_size=hparams['data_image_resize_size'], test=True)

        test_loader = DataLoader(
            test_dataset, batch_size=hparams['data_batch_size'], shuffle=hparams['data_shuffle'], num_workers=hparams['data_num_workers'], pin_memory=hparams['data_pin_memory'], drop_last=True)

        return test_loader


if __name__ == '__main__':
    # basic test script
    root_src = './'
    data_src = root_src + 'cityscapes'

    dataset = Cityscapes_Train(
        data_src, resize_size=224)
    print(len(dataset))

    dataset = Cityscapes_Val(
        data_src, resize_size=224)
    print(len(dataset))

    dataset = Cityscapes_Test(
        data_src, resize_size=224, test=True)
    print(len(dataset))


