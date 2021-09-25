import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image as im
from scipy.ndimage import interpolation as inter



#import image
image = cv2.imread('exemplos/exemplo8.jpeg')
image = cv2.resize(image,(1024,1048))
cv2.imshow('orig',image)
# cv2.waitKey(0)

dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15) 
# Plotting of source and destination image 


grayDST = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray)
# cv2.waitKey(0)
#threshold

# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6) #imgf contains Binary image
# cv2.imshow('thresh',thresh)

threshDST = cv2.adaptiveThreshold(grayDST,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6) #imgf contains Binary image
cv2.imshow('threshDST',threshDST)

# img = cv2.imread('j.png',0)
# kernel = np.ones((1,1),np.uint8)
# erosion = cv2.erode(threshDST,kernel,iterations = 1)
# cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()