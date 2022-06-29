import os
import pytorch_lightning as pl
import torch
from transformers import Data2VecVisionForSemanticSegmentation, Data2VecVisionConfig
from torch import nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW, lr_scheduler
import pathlib
from torchvision import transforms


class Data2VecLightning(pl.LightningModule):

    def __init__(self, save_location, hparams,pretrained_location=None):
        super(Data2VecLightning, self).__init__()
        self.hparams.update(hparams)
        self.hparams.save_location = save_location
        self.hparams.initial_learn_rate = 0.001
        self.hparams.post_BEIT_output_resize_mode = 'bilinear'
        self.hparams.adam_epsilon = 1e-3
        self.hparams.reduce_lr_on_plateau_mode = 'min'
        

        config = Data2VecVisionConfig('facebook/data2vec-vision-base-ft1k')
        config.num_labels = self.hparams.data_num_classes
        config.image_size = self.hparams.data_image_resize_size
        if pretrained_location is not None:
            model = Data2VecVisionForSemanticSegmentation.from_pretrained(pretrained_location)
        else:
            model = Data2VecVisionForSemanticSegmentation.from_pretrained(
                'facebook/data2vec-vision-base-ft1k', config=config)
        self.model = model
        self.data_src = self.hparams.data_data_src

        self.save_hyperparameters()
        self.convertTensorToPilImage = transforms.ToPILImage()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.hparams.initial_learn_rate,
                          eps=self.hparams.adam_epsilon,
                          )

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.hparams.reduce_lr_on_plateau_mode,
        )

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", 'name': 'lr_scheduler'}
                }

    @torch.autocast(device_type='cuda')
    def forward(self, pixel_values, labels):
        return self.model(pixel_values=pixel_values, labels=labels)

    def pre_evaluation_step(self, outputs, labels):
        upsampled_logits = self.upsample(outputs['logits'])
        predicted = upsampled_logits.argmax(dim=1)
        # we don't include the background class in the accuracy calculation
        mask = (labels != 255)
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()

        return pred_labels, true_labels

    def upsample(self, outputs):

        upsampled_logits = nn.functional.interpolate(
            outputs, size=[self.hparams.data_image_resize_size, self.hparams.data_image_resize_size], mode=self.hparams.post_BEIT_output_resize_mode,)

        return upsampled_logits

    def training_step(self, batch, batch_idx):
        images, img_batch_paths, labels, label_batch_paths = batch

        outputs = self.forward(images, labels)
        pred_labels, true_labels = self.pre_evaluation_step(outputs, labels)
        accuracy = accuracy_score(pred_labels, true_labels)

        loss = outputs['loss']
        self.log('train_pixel_accuracy', accuracy,
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_loss', loss,
                 logger=True, on_step=True, on_epoch=True)
        return {'loss': loss, 'prediction': outputs['logits'], 'ground_truth': labels}

    def validation_step(self, batch, batch_idx):
        images, img_batch_paths, labels, label_batch_paths = batch
        outputs = self.forward(images, labels)
        pred_labels, true_labels = self.pre_evaluation_step(outputs, labels)
        accuracy = accuracy_score(pred_labels, true_labels)
        loss = outputs['loss']
        self.log('val_pixel_accuracy', accuracy,
                 logger=True, on_step=True, on_epoch=True)
        self.log('val_loss', loss,
                 logger=True, on_step=True, on_epoch=True)

        upsampled_predicted_pixels = self.upsample(outputs['logits'])
        upsampled_predicted_pixels = upsampled_predicted_pixels.argmax(dim=1)

        sub_folder = os.path.join('val_outputs', str(self.global_step))

        self.saveImage(
            label_batch_paths, upsampled_predicted_pixels, sub_folder)

        self.saveImage(
            img_batch_paths, images, sub_folder)

        return {'val_loss': loss, 'val_pixel_accuracy': accuracy}

    def test_step(self, batch, batch_idx):
        images, paths = batch
        outputs = self.forward(images, labels=None)
        predicted_pixels = outputs['logits']
        upsampled_predicted_pixels = self.upsample(predicted_pixels)
        upsampled_predicted_pixels = upsampled_predicted_pixels.argmax(dim=1)

        sub_folder = 'test_outputs'
        self.saveImage(paths, upsampled_predicted_pixels, sub_folder)

    def saveImage(self, paths, upsampled_predicted_pixels, sub_folder):

        for (predicted_pixels, path) in zip(upsampled_predicted_pixels[:], paths[:]):
            save_prediction_path = self.generateImageSavePath(sub_folder, path)
            predicted_pixels = predicted_pixels.type(torch.uint8)
            img = self.convertTensorToPilImage(predicted_pixels)
            img.save(save_prediction_path, format='PNG')

    def generateImageSavePath(self, sub_folder, path):
        path = pathlib.Path(path)
        relative_path = path.relative_to(self.data_src)
        save_prediction_path = os.path.join(
            self.hparams.save_location, sub_folder, relative_path)
        save_prediction_path = pathlib.Path(save_prediction_path)
        os.makedirs(save_prediction_path.parent, exist_ok=True)
        # ugly could be done better
        save_prediction_path = '{}{}'.format(
            save_prediction_path.__str__().split('.')[0], '.png')

        return save_prediction_path

    def on_train_end(self):
        
        self.model.save_pretrained(os.path.join(
            self.hparams.save_location, 'Data2VecLightning_patch16_224_epoch_{}_step_{}'.format(self.current_epoch, self.global_step)))
        


if __name__ == '__main__':
    data_hparams = {
        'data_data_src': './cityscapes',
        'data_batch_size': 75,
        'data_image_resize_size': 512,
        'data_num_workers': 4,
        'data_persistent_workers': True,
        'data_pin_memory': True,
        'data_shuffle': True,
        # Number of classes (inc 0 and 255, 18 are actaully classes)
        'data_num_classes': 20,
    }
    module = Data2VecLightning(save_location='./asdf', hparams=data_hparams)
