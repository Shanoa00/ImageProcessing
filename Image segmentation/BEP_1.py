import cv2 as cv
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

def read_img():
    lena = cv.imread("lena.jpg") # Read Img in BGR Colour
    cv.imshow("Color", lena) # Show Img
    #print("Original Size: ", lena.shape)
    #print("pixel value [10,10]: ", lena[10,10])
    return lena #,lena_gray

def green():
    lena = read_img()
    print("Data type of Img : ", lena.dtype)
    green = np.zeros((225,225,3), dtype=np.uint8)
    for i in range(225):
        for j in range(225):
            green[i,j,1] = lena[i,j,1]
    cv.imshow("Green", green)
    print(green[10,10,1])
    return green


#Assaigment 1:----------------------------------
def my_gray():
    lena = read_img()
    gray= np.zeros((225,225,3), dtype=np.uint8)
    b, g, r = lena[:,:,0], lena[:,:,1], lena[:,:,2]
    avg = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #print(avg.shape)
    for i in range(3):
        gray[:,:,i]= avg
    cv.imshow("My_Gray", gray)
    print("Random pixel value (My gray): ", gray[10,83,0])
    #-----Comparing with function---
    #lena_gray = cv.imread("lena.jpg", cv.IMREAD_GRAYSCALE)
    #cv.imshow("Original_Gray", lena_gray)
    #print("Same pixel value (Original gray): ", lena_gray[10,83])

    return gray

def draw_circle(x,y):
    img = read_img()
    img = cv.circle(img, (x,y), 20, (255,0,0), 2)
    cv.imshow("Circle", img)
    return img

#Assigment 2: Histogram----------------------------
def histogram():
    gray= my_gray()
    count= Counter(gray[:,:,0].flatten())
    print(count.most_common())
    plt.bar(count.keys(), count.values())
    plt.title('Histogram')
    plt.show()

def main():
    #green()
    #Asigment 1: 
    #my_gray()
    #draw_circle(148,115)

    #Asigment 2:
    histogram()

if __name__=="__main__":
    main()
    cv.waitKey(0)

