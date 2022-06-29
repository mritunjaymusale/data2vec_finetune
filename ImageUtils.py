from PIL import Image
from torchvision.transforms.functional import normalize
import numpy as np
import torch
from PIL import ImageTk
import cv2


class ImagePreprocessing(object):
    @staticmethod
    def read_image(path, is_greyscale=False):
        if is_greyscale:
            image = Image.open(path).convert('L')
        else:
            image = Image.open(path).convert('RGB')
        return image

    @staticmethod
    def resizeImg(image, size: int):
        image = image.resize((size, size), Image.Resampling.NEAREST)

        return image

    @staticmethod
    def convertImgToTensor(image, is_greyscale=False):
        image = np.array(image)
        # for masks, the output will be 3d tensor and long dtype
        if is_greyscale:
            image = torch.from_numpy(image)
            image = image.long()

        else:
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            image = image.float()

        return image

    @staticmethod
    def normalize_image(image):
        image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image





# for dectectron2
class ImageHandler():

    @staticmethod
    def convertToTkImage(image, ):
        image = ImageTk.PhotoImage(image)

        return image

    @staticmethod
    def arrrayToImage(image, segmented_output):
        image = Image.fromarray(image)
        segmented_output = Image.fromarray(segmented_output)
        return image, segmented_output

    @staticmethod
    def resizeImages(image, segmented_output):
        image.thumbnail((720, 720), Image.ANTIALIAS)
        segmented_output.thumbnail((720, 720), Image.ANTIALIAS)

    @staticmethod
    def openCVtoNormal(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':

    mask = ImagePreprocessing.read_image(
        'IDD/idd20kII/gtFine/train/201/frame0029_gtFine_labellevel3Ids.png', is_greyscale=True)
    mask = ImagePreprocessing.resizeImg(mask, 256)
    mask = ImagePreprocessing.convertImgToTensor(mask, is_greyscale=True)
    print(mask)