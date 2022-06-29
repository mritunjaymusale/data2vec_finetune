from logging import root
from typing import List
from PIL import Image
import yaml
from dataset_metadata import CityScapesLabel, CityScapesLabels, IDDLabel, IDDLabels, getNamesandColors
from lightning_module import Data2VecLightning
from ImageUtils import ImageHandler, ImagePreprocessing
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from tkinter import filedialog
from tkinter import *
from PIL import Image
import numpy as np
import torch
from torch import nn
from detectron2.data import MetadataCatalog, DatasetCatalog


class SSLDemo(object):

    generated_masks = None
    path = None
    root = Tk()
    panelA = None
    panelB = None

    def __init__(self):
        with open('./config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        self.idd_config = config['idd']['data_hparams']
        self.cityscapes_config = config['cityscapes']['data_hparams']
        self.resize_size = 224

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        text = Label(self.root, text="Select model")
        text.pack()
        btn = Button(self.root, text="Cityscapes",
                     command=lambda: self.cityscapesModel()
                     )
        btn.pack(side="bottom", fill="both",
                 expand="yes", padx="10", pady="10")
        btn2 = Button(self.root, text="IDD",
                      command=lambda: self.iddModel()
                      )
        btn2.pack(side="bottom", fill="both",
                  expand="yes", padx="10", pady="10")

    def setMetadataGlobals(self):
        if self.current_dataset_type == 'idd':
            class_names, class_colors = getNamesandColors(
                IDDLabel, IDDLabels, 'idd')
            DatasetCatalog.register("idd", lambda: List[dict])
            MetadataCatalog.get("idd").stuff_classes = class_names
            MetadataCatalog.get("idd").stuff_colors = class_colors
        else:
            class_names, class_colors = getNamesandColors(
                CityScapesLabel, CityScapesLabels, 'cityscapes')
            DatasetCatalog.register("cityscapes", lambda: List[dict])
            MetadataCatalog.get("cityscapes").stuff_classes = class_names
            MetadataCatalog.get("cityscapes").stuff_colors = class_colors

    def cityscapesModel(self):
        self.current_dataset_type = 'cityscapes'
        self.model = self.initializeModel()
        self.setMetadataGlobals()
        self.root.destroy()
        self.root = Tk()
        self.root.title("Cityscapes")
        self.root.geometry("200x100")
        btn = Button(self.root, text="Select an image",
                     command=lambda: self.imageSelectionAndSegmentation())
        btn.pack(side="bottom", fill="both",
                 expand="yes", padx="10", pady="10")

    def iddModel(self):
        self.current_dataset_type = 'idd'
        self.model = self.initializeModel()
        self.setMetadataGlobals()
        self.root.destroy()
        self.root = Tk()
        self.root.title("IDD")
        self.root.geometry("200x100")
        btn = Button(self.root, text="Select an image",
                     command=lambda: self.imageSelectionAndSegmentation())
        btn.pack(side="bottom", fill="both",
                 expand="yes", padx="10", pady="10")

    def imageSelectionAndSegmentation(self):

        self.path = filedialog.askopenfilename()

        if len(self.path) > 0:

            resized_image, transformer_output = self.segmentUsingData2Vec()
            cfg, predictor = self.initializeDetectronPredictorAndConfig()

            # output of the mask-RCNN
            outputs = predictor(np.array(resized_image))

            v = Visualizer(
                resized_image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
            detectron_output = v.draw_instance_predictions(
                outputs["instances"].to("cpu"))

            self.updateGUI(np.array(resized_image),
                           transformer_output.get_image(), detectron_output.get_image())

    def segmentUsingData2Vec(self):
        image = ImagePreprocessing.read_image(self.path)
        resized_image = ImagePreprocessing.resizeImg(
            image, self.resize_size)
        image = ImagePreprocessing.convertImgToTensor(resized_image)
        image = image.unsqueeze(0)

        outputs = self.model(image, labels=None)
        upsampled_logits = self.upsample(outputs['logits'])
        predicted = upsampled_logits.argmax(dim=1)
        predicted = predicted.squeeze(0)

        v = Visualizer(resized_image, metadata=MetadataCatalog.get(
            self.current_dataset_type))

        out = v.draw_sem_seg(predicted)
        return resized_image, out

    def upsample(self, outputs):
        if self.current_dataset_type == 'cityscapes':
            upsampled_logits = nn.functional.interpolate(
                outputs, size=[self.cityscapes_config['data_image_resize_size'], self.cityscapes_config['data_image_resize_size']], mode='nearest',)
        else:
            upsampled_logits = nn.functional.interpolate(
                outputs, size=[self.idd_config['data_image_resize_size'], self.idd_config['data_image_resize_size']], mode='nearest',)

        return upsampled_logits

    def updateGUI(self, image, transformer_output, detectron_output):

        image = Image.fromarray(image)
        transformer_output = Image.fromarray(transformer_output)
        detectron_output = Image.fromarray(detectron_output)

        image = ImagePreprocessing.resizeImg(image, 720)
        transformer_output = ImagePreprocessing.resizeImg(transformer_output, 720)
        detectron_output = ImagePreprocessing.resizeImg(detectron_output, 720)

        image = ImageHandler.convertToTkImage(
            image)
        transformer_output = ImageHandler.convertToTkImage(
            transformer_output)
        detectron_output = ImageHandler.convertToTkImage(
            detectron_output)

        if self.panelA is None or self.panelB is None or self.panelC is None:
            self.initializePanels(image, transformer_output, detectron_output)
        else:
            self.updatePanels(image, transformer_output, detectron_output)

    def initializeModel(self,):
        model = None
        if self.current_dataset_type == 'cityscapes':
            model = Data2VecLightning(
                None, self.cityscapes_config, pretrained_location='demo_data/cityscapes_demo/Data2Vec_Test13_saved_model')

        elif self.current_dataset_type == 'idd':
            model = Data2VecLightning(
                None, self.idd_config, pretrained_location='demo_data/IDD_demo/Data2Vec_Test12_saved_model')

        model.eval()
        return model

    def initializeDetectronPredictorAndConfig(self):
        cfg = get_cfg()
        cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        return cfg, predictor

    def initializePanels(self, image, transformer_output, detectron_output):

        # the first panel will store the original image

        self.panelA = Label(self.root, image=image, text="Original Image")
        self.panelA.image = image
        self.panelA.pack(side="left", padx=10, pady=10)

        # the second panel will store the masked output
        self.panelB = Label(
            self.root, image=transformer_output, text="Transformer Segmented Image")
        self.panelB.image = transformer_output
        self.panelB.pack(side="left", padx=10, pady=10)

        # the third panel will store the detectron output
        self.panelC = Label(
            self.root, image=detectron_output, text="Detectron Segmented Image")
        self.panelC.image = detectron_output
        self.panelC.pack(side="right", padx=10, pady=10)

    def updatePanels(self, image, transformer_output, detectron_output):

        self.panelA.configure(image=image)
        self.panelB.configure(image=transformer_output)
        self.panelC.configure(image=detectron_output)
        self.panelA.image = image
        self.panelB.image = transformer_output
        self.panelC.image = detectron_output


demoApp = SSLDemo()


demoApp.root.mainloop()
