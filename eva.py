import keras
import os
from deepsense import neptune
from keras_retinanet.bin.train import create_models
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.resnet import download_imagenet, resnet_retinanet as retinanet
from keras_retinanet.preprocessing.detgen import DetDataGenerator
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.preprocessing.detgen import DetDataGenerator
from detdata import DetGen
from detdata.augmenters import crazy_augmenter
from keras_retinanet.utils.eval import evaluate

import warnings
warnings.filterwarnings("ignore")

ctx = neptune.Context()

detg= DetGen('/home/i008/malaria_data/dataset_train.mxrecords',
      '/home/i008/malaria_data/dataset_train.csv',
      '/home/i008/malaria_data/dataset_train.mxindex', batch_size=4)


train_generator = DetDataGenerator(detg, augmenter=crazy_augmenter)
train_generator.image_max_side = 750
train_generator.image_min_side = 750




weights = download_imagenet('resnet50')

model_checkpoint = keras.callbacks.ModelCheckpoint('mod-{epoch:02d}_loss-{loss:.4f}.h5',
                                                   monitor='loss',
                                                   verbose=2,
                                                   save_best_only=False,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)

callbacks = []


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        ctx.channel_send("loss", logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        ctx.channel_send("val_loss", logs.get('val_loss'))


callbacks.append(keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=2,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=0,
    min_lr=0
))

# callbacks.append(LossHistory())
callbacks.append(model_checkpoint)

model, training_model, prediction_model = create_models(
    backbone_retinanet=retinanet,
    backbone='resnet50',
    num_classes=train_generator.num_classes(),
    weights=weights,
    multi_gpu=0,
    freeze_backbone=True
)

model.load_weights('mod-40_loss-2.0113.h5')
g = CSVGenerator('csviterval.csv', 'cls.csv')
average_precisions, recall, precision, true_positives, false_positives = evaluate(g, model, score_threshold=0.1)
