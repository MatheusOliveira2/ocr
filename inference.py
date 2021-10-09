import cv2
import numpy as np
import os
import shutil
import sys
import tensorflow as tf
import math
from tensorflow.keras.models import load_model

model_dir = '/Users/CromAI/Documents/ocr/ocr/models/segmentador-epoch=17-acc=0.883-val_acc=0.875.model'
image_path = '/Users/CromAI/Documents/ocr/ocr/exemplos/exemplo5.jpeg'
filename = image_path.split('/')[-1]
size = 100
model = load_model(model_dir)



img = cv2.imread(image_path)     
altura = img.shape[0]
largura = img.shape[1]
scan = np.ones((altura,largura,1),np.uint8)
k = math.ceil(altura/size)
l = math.ceil(largura/size)
i=0
j=0
while(i<k-1):
    while(j<l-1):
        crop = img[(size*i):((size*i)+size), (size*j):((size*j)+size)]

        input = (np.expand_dims(crop/255,0))

        prediction = model.predict(input)
        output = np.squeeze(prediction,0)
        output=output*255
                 
        scan[(size*i):((size*i)+size), (size*j):((size*j)+size)] = output
        j=j+1
        
    i=i+1
    j=0

cv2.imwrite('inferece_'+filename, scan)
