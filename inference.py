import cv2
import numpy as np
import math
from tensorflow.keras.models import load_model
import os
from time import sleep
model_dir = '/Users/CromAI/Documents/ocr/ocr/models/segmentador-epoch=47-acc=0.107-val_acc=0.105.model'#segmentador-epoch=35-acc=0.107-val_acc=0.105.model'#segmentador-epoch=49-acc=0.583-val_acc=0.584.model'
#image_path = '/Users/CromAI/Documents/ocr/ocr/exemplos/recortes_e_filtros/exemplo16.jpeg'
list = os.listdir('/Users/CromAI/Documents/ocr/teste')


def do_inferece(image_path, model_dir):
    filename = image_path.split('/')[-1]
    size = 100
    model = load_model(model_dir)


    img = cv2.imread(image_path)     
    altura = img.shape[0]
    largura = img.shape[1]
    k = math.ceil(altura/size)
    l = math.ceil(largura/size)

    img_normalizada = np.zeros((k*size, l*size,3), dtype=img.dtype)
    img_normalizada[:] = (0,0,0)
    img_normalizada[:img.shape[0],:img.shape[1]] = img
    cv2.imwrite(image_path,img_normalizada)

    scan = np.zeros((k*size, l*size,1), dtype=img.dtype)
    scan[:] = 0
    i=0
    j=0
    while(i<2*(k-1)):
        while(j<2*(l-1)):
            crop = img[(int(size/2)*i):((int(size/2)*i)+size), (int(size/2)*j):((int(size/2)*j)+size)]
            if (crop.shape[0] != size) or (crop.shape[1] != size):
                norm_crop = np.zeros((size, size,3), dtype=crop.dtype)
                norm_crop[:] = (0,0,0)
                norm_crop[:crop.shape[0],:crop.shape[1]] = crop
                crop = norm_crop

            input = (np.expand_dims(crop/255,0))

            prediction = model.predict(input)
            output = np.squeeze(prediction,0)
            output=output*255
            #cv2.imwrite('output.jpg',255-output)
            #cv2.imwrite('bkg.jpg', (scan[(int(size/2)*i):((int(size/2)*i)+output.shape[0]), (int(size/2)*j):((int(size/2)*j)+output.shape[1])]))
            soma = cv2.addWeighted((scan[(int(size/2)*i):((int(size/2)*i)+output.shape[0]), (int(size/2)*j):((int(size/2)*j)+output.shape[1])]),
                                    1,
                                    255-(np.array(output, dtype=img.dtype)),
                                    1,
                                    0)
            scan[(int(size/2)*i):((int(size/2)*i)+output.shape[0]), (int(size/2)*j):((int(size/2)*j)+output.shape[1])] = np.expand_dims(soma,2)
            #cv2.imwrite('soma.jpg', soma)
            #cv2.imwrite('teate.jpg', scan)
            #sleep(1)
            j=j+1
            
        i=i+1
        j=0
    scan = cv2.bitwise_not(scan)
    scan = cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR)
    scan = cv2.fastNlMeansDenoisingColored(scan, None, 10, 10, 3, 15)
    cv2.imwrite('/Users/CromAI/Documents/ocr/teste/'+filename.replace('.jpg','_inference.jpeg'), scan)

for imagem in list:
    print(imagem)
    if imagem.endswith('.jpeg') or imagem.endswith('.jpg') or imagem.endswith('.png'):
        image_path = '/Users/CromAI/Documents/ocr/teste/'+imagem
        do_inferece(image_path,model_dir)