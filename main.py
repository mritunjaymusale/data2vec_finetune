import os
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from IDD import IDDDataLoaders
from cityscapes import CityscapesDataLoaders
from lightning_module import Data2VecLightning
import yaml
pl.seed_everything(000, workers=True)


root_src = '/mnt/HDD'
save_location = os.path.join(root_src, 'results',
                             datetime.today().strftime('%Y-%m-%d'),
                             datetime.today().strftime('%H:%M:%S'),
                             )
log_location = os.path.join(save_location, 'logs',)

if not os.path.exists(log_location):
    os.makedirs(log_location)

tf_logger = TensorBoardLogger(save_dir=log_location,
                              name="Data2Vec", version=4)
csv_logger = pl.loggers.CSVLogger(os.path.join(log_location, 'csv_logger'))


amp_plugin = NativeMixedPrecisionPlugin(precision=16, device='cuda')

lr_monitor = LearningRateMonitor(logging_interval='step')


with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)


idd_config = config['idd']['data_hparams']
cityscapes_config = config['cityscapes']['data_hparams']
trainer_config = config['trainer_settings']

# change this based on the needed dataset
data_hparams = idd_config

train_data, val_data, test_data = None, None, None
if data_hparams['data_data_src'] == './IDD':
    train_data = IDDDataLoaders.get_train_dataloader(data_hparams)
    val_data = IDDDataLoaders.get_val_dataloader(data_hparams)
    test_data = IDDDataLoaders.get_test_dataloader(data_hparams)
else:
    train_data = CityscapesDataLoaders.get_train_dataloader(data_hparams)
    val_data = CityscapesDataLoaders.get_val_dataloader(data_hparams)
    test_data = CityscapesDataLoaders.get_test_dataloader(data_hparams)
logging_interval = 5  # Logging interval


trainer = pl.Trainer(
    **trainer_config,
    callbacks=[lr_monitor],
    plugins=[amp_plugin],
    logger=[tf_logger, csv_logger],
)


model = Data2VecLightning(save_location, data_hparams)


trainer.fit(model, train_data, val_data)

# this may not work use the model saved internally in the trainer instead check on_train_end() from lightning_module.py
trainer.save_checkpoint(os.path.join(save_location, 'checkpoint.ckpt'))

trainer.test(model, test_data)
