import numpy as np
import cv2
import os 
import math
size = 100
list = os.listdir('/Users/CromAI/Documents/ocr/ocr/exemplos/cropagem/ocr')
cont = 1
for imagem in list:
    print(imagem)
    if imagem.endswith('.jpg'):
        img = cv2.imread('/Users/CromAI/Documents/ocr/ocr/exemplos/cropagem/ocr/'+imagem)
        
        altura = img.shape[0]
        largura = img.shape[1]

        k = math.ceil(altura/size)
        l = math.ceil(largura/size)
        i=0
        j=0
        print(imagem)
        while(i<k-1):
            
            while(j<l-1):
                crop = img[(size*i):((size*i)+size), (size*j):((size*j)+size)]

                cv2.imwrite('/Users/CromAI/Documents/ocr/ocr/exemplos/cropagem/crops/'+str(cont)+'.jpg', crop)
                
                cont=cont+1
                j=j+1
                

            i=i+1
            j=0