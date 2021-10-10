import cv2

img  = cv2.imread("binary.png")

height, width = img.shape[:2]

for i in range(0, height-150):
    for j in range(0, width):
        img[i,j] = [0,0,0]

cv2.imshow("binary", img)
cv2.waitKey()