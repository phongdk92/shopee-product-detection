#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 30 18:21 2020

@author: phongdk
"""

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fastai.vision import *
from efficientnet_pytorch import EfficientNet


DATA_DIR = '/home/phongdk/shopee/'
train_path = f'{DATA_DIR}/train/train/'
test_path = f'{DATA_DIR}/test/test/'
data = ImageDataBunch.from_folder(train_path,
                                  ds_tfms=get_transforms(),
                                  size=(224, 224),
                                  valid_pct=0.2,
                                  seed=42,
                                  bs=64).normalize(imagenet_stats)

model = EfficientNet.from_name('efficientnet-b0')

for param in model.parameters():
    param.requires_grad = False
criterion = nn.NLLLoss()

learn = Learner(data, model, loss_func=criterion, metrics=accuracy).mixup()

learn.model_dir = os.getcwd()   # set path to save model
learn.fit_one_cycle(1)

# learn.save('baseline.model')

# il = ImageList.from_folder(test_path).databunch().normalize(imagenet_stats)
# for img in il:
#     prediction = learn.predict(img)

# submission = pd.read_csv(f"{DATA_DIR}/test.csv")
# submission['category'] = submission['filename'].apply(lambda x: learn.predict())
# models.resnet152()
