import cv2 as cv
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import statistics

def read_img(image):
    try:
         img = cv.imread(image) # Read Img in BGR Colour
    except:
        print("check image path!!")
    #cv.imshow("Color", lena) # Show Img
    #print("Original Size: ", lena.shape)
    #print("pixel value [10,10]: ", lena[10,10])
    return img #,lena_gray

def green():
    try:
        lena = read_img("lena.jpg")
    except:
        print("Check image path!")
    print("Data type of Img : ", lena.dtype)
    green = np.zeros((225,225,3), dtype=np.uint8)
    for i in range(225):
        for j in range(225):
            green[i,j,1] = lena[i,j,1]
    cv.imshow("Green", green)
    print(green[10,10,1])
    return green


#Assaigment 1:----------------------------------
def my_gray(image):
    try:
        lena = read_img(image)
    except:
        print("Check image path!")
    gray= np.zeros((lena.shape[0],lena.shape[1],3), dtype= np.uint8)
    b, g, r = lena[:,:,0], lena[:,:,1], lena[:,:,2]
    avg = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #print(avg.shape)
    for i in range(3):
        gray[:,:,i]= avg
    cv.imshow("My_Gray", gray)
    #print("Random pixel value (My gray): ", gray[10,83,0])
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
def histogram(image):
    gray= my_gray(image)
    #print(gray.shape)
    count= Counter(gray[:,:,0].flatten())
    #print(count.most_common())
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.bar(count.keys(), count.values())
    fig.suptitle('Histogram')
    
    ##sliding:
    offset= 35
    of_img= gray[:,:,0]+ offset
    ##Set max & min limits: 
    of_img = np.clip(of_img,0,255)
    cv.imshow("Offset", of_img)
    count_off= Counter(of_img.flatten())
    ax2.bar(count_off.keys(), count_off.values())
    plt.title('Histogram_sliding')
    plt.show()

def stretching():
    gray= my_gray()
    print(gray.shape)
    stret_img = np.zeros((gray.shape[0],gray.shape[1]),dtype = 'uint8')
    # Loop over the image and apply Min-Max formulae
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            stret_img[i,j] = 255*(gray[i,j,0]-np.min(gray))/(np.max(gray)-np.min(gray))
    cv.imshow("Stretching", stret_img)

    count= Counter(gray[:,:,0].flatten())
    #print(count.most_common())
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.bar(count.keys(), count.values())
    fig.suptitle('Histogram')
    count_off= Counter(stret_img.flatten())
    ax2.bar(count_off.keys(), count_off.values())
    plt.title('Histogram_stretching')
    plt.show()


def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) 
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = (norm / n_)*255
    uniform_norm =  uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq

def equalize(image):
    try:
        image_src = my_gray(image)
    except:
        print("Check image path!")
    image_eq = enhance_contrast(image_matrix=image_src)
    cv.imshow("equalize", image_eq.astype(np.uint8))
    count= Counter(image_src[:,:,0].flatten())
    #print(count.most_common())
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.bar(count.keys(), count.values())
    fig.suptitle('Histogram')
    count_off= Counter(image_eq.flatten())
    ax2.bar(count_off.keys(), count_off.values())
    plt.title('Histogram_equialized')
    plt.show()


def threshold(image, thres): 
    try:
        image_src = cv.imread(image,0)
    except:
        print("Check image path!")
    thres_img =  np.zeros((image_src.shape[0],image_src.shape[1]), dtype= np.uint8)
    for i in range(image_src.shape[0]):
        for j in range(image_src.shape[1]):
            if image_src[i,j]>thres:
                thres_img[i,j]= 1
    #print(thres_img[100])
    fig=plt.figure()
    fig.add_subplot(1, 2, 1)   # subplot one
    plt.imshow(image_src, cmap=plt.cm.gray)
    fig.add_subplot(1, 2, 2)   # subplot two
    # my data is OK to use gray colormap (0:black, 1:white)
    plt.imshow(thres_img, cmap=plt.cm.gray)  # use appropriate colormap here
    plt.show()

def dynamic_tres(image):
    try:
        img = cv.imread(image,0)
    except:
        print("Check image path!")
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    pdf = cv.calcHist([img], [0], None, [256], [0, 256])
    #print(pdf)
    plt.plot(pdf)
    # plt.hist(img.ravel(),256,[0,256])
    
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.plot(cdf * hist.max() / cdf.max(), color='b')
    #plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    print()
    
def iterative(imgObj, T0=100, maxIter=200):
        
        img  = cv.imread(imgObj,0)

        T = T0
        u1 = np.array(img[img <= T], dtype=np.float16).mean()
        print(np.array(img[img <= T], dtype=np.float16).mean())
        print((np.array(img[img <= T], dtype=np.float16).shape))
        for i in range(0, maxIter):
            u1 = np.array(img[img <= T], dtype=np.float16).mean()
            u2 = np.array(img[img > T], dtype=np.float16).mean()
            Tnew = (u1 + u2) / 2
            
            if abs(Tnew - T) < 0.01:
                break
            else:
                T = Tnew
                #print(T)
        



def main():
    #green()
    #Asigment 1: 
    #my_gray()
    #draw_circle(148,115)

    #Asigment 2:
    #histogram()
    #stretching()
    #equalize("landscape.jpg")
    #threshold("lena.jpg",100)
    #dynamic_tres("lena.jpg")
    iterative("lena.jpg")

if __name__=="__main__":
    main()
    cv.waitKey(0)

