import cv2
from matplotlib import pyplot as plt
import numpy as np
import statistics
from peakdetect import peakdetect

def line_segmentation(image):
    height = image.shape[0]-200
    width = image.shape[1]
    print ("height: ", height)
    print ("width: ", width)
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
    histogram(array)
    find_line(image,array)

def histogram(array):
    ax2 = plt.subplot(2, 1, 2)
    peaks = peakdetect(array, lookahead=20)
    higherPeaks = np.array(peaks[1])
    ax2.plot(higherPeaks[:,0], higherPeaks[:,1], 'bx')
    ax2.bar(range(len(array)), array, color='r')
    # plt.show()
    plt.savefig("histogram-line5.png")

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

    cv2.imshow("lines", imageColor)
    cv2.waitKey()
    cv2.destroyAllWindows()


image = cv2.imread('exemplo9.jpeg', 0)
image = cv2.bitwise_not(image)
kernel = np.ones((2,2), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)
image = cv2.bitwise_not(image)
line_segmentation(image)

    