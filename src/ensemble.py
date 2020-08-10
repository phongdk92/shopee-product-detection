#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 28 22:14 2020

@author: phongdk
"""

from fastai.vision import *
from fastai.metrics import error_rate
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = '/home/phongdk/shopee/'
train_path = f'{DATA_DIR}/train/train/'
test_path = f'{DATA_DIR}/test/test/'
il = ImageList.from_folder(test_path)
data = ImageDataBunch.from_folder(train_path,
                                  ds_tfms=get_transforms(),
                                  size=224,
                                  valid_pct=0.2,
                                  seed=42)


learn = cnn_learner(data, models.densenet201, metrics=accuracy).mixup()
learn.model_dir = os.getcwd()
learn.data.batch_size = 16
learn.load('densenet201_mixup.model')
predicts = []
for image in tqdm(il):
    predicts.append(to_np(learn.predict(image)[-1]))

del learn

learn = cnn_learner(data, models.resnet152, metrics=accuracy).mixup()
learn.model_dir = os.getcwd()
learn.data.batch_size = 16
learn.load('resnet152_mixup.model')
predicts2 = []
for image in tqdm(il):
    predicts2.append(to_np(learn.predict(image)[-1]))

predicts = np.stack(predicts)
predicts2 = np.stack(predicts2)
w = 0.55
predicts = np.array(predicts) * w + np.array(predicts2) * (1-w)

sub = pd.DataFrame({'filename': il.items, 'category': predicts})
sub['filename'] = sub['filename'].apply(lambda x: str(x).split('/')[-1])
submission = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = submission.merge(sub, how='left', on='filename')
submission['category'] = submission['category_y']
submission[['filename', 'category']].to_csv('ensemble.csv', index=False)
