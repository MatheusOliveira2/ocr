import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from numpy.lib.type_check import imag
from scipy.ndimage import interpolation as inter
import statistics
from peakdetect import peakdetect

def threshold(image):
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15) 
    grayDST = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    threshDST = cv2.adaptiveThreshold(grayDST,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6) #imgf contains Binary image
    return threshDST
    # cv2.imshow('threshDST',threshDST)
    # cv2.imwrite('exemplo6.jpg',threshDST)
    # # kernel = np.ones((1,1),np.uint8)
    # # erosion = cv2.erode(threshDST,kernel,iterations = 1)
    # # cv2.imshow('erosion',erosion)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def skewCorrection(image):
    img = im.fromarray(image)
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    # plt.imshow(bin_img, cmap='gray')
    # plt.savefig('binary.png')
    
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        # print('Best angle: {}'.format(best_angle))
        data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
        img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
        # img.save('skew_corrected.png')

    return np.array(img)


def line_segmentation(image, index):
    height = image.shape[0]
    width = image.shape[1]
    # print ("height: ", height)
    # print ("width: ", width)
    array = []

    blackPixel = 0
    for i in range(height):
        for j in range(width):
            if image[i,j] < 127:
                blackPixel += 1
            if(j == width-1):
                array.append(blackPixel)
                blackPixel = 0

    # array = normalizeData(array, width)
    histogram(array, index)
    image = find_line(image,array)
    return image

def histogram(array, index):
    ax2 = plt.subplot(2, 1, 2)
    peaks = peakdetect(array, lookahead=20)
    higherPeaks = np.array(peaks[1])
    ax2.plot(higherPeaks[:,0], higherPeaks[:,1], 'bx')
    ax2.bar(range(len(array)), array, color='r')
    # plt.show()
    plt.savefig("histogram-line" + str(index) + ".png")

def normalizeData(array,width):
    for i in range(len(array)):
        array[i] = array[i]/width
    return array

def find_line(image,array):
    mean = statistics.mean(array)
    print ("mean: ", mean)
    
    width = image.shape[1]
    lines = []
    # for a in range(len(array)):
    #     if array[a] <= mean and array[a] > mean/2:
    #         lines.append(a)

    peaks = peakdetect(array, lookahead=20)
    higherPeaks = np.array(peaks[1])

    for peak in higherPeaks:
            lines.append(peak)

    print ("lines: ", len(lines))
    
    imageColor = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

    
    for i in range(len(lines)):
        for j in range(width):
            imageColor[lines[i],j] = [0,0,255,40]

    # cv2.imshow("lines", imageColor)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return imageColor

for i in range(1,11):
    image_path = 'exemplos/exemplo' + str(i) +'.jpeg'
    image = cv2.imread(image_path)
    image = threshold(image)
    image = cv2.cvtColor(skewCorrection(image), cv2.COLOR_RGB2GRAY)
    image = cv2.bitwise_not(image)
    image = line_segmentation(image,i)

    cv2.imwrite("find_line" + str(i) + ".jpeg", image)
