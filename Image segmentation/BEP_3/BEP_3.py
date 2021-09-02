import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.records import array

def avg_filter(image):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
    except:
        print("check image path!!")  
    mask = np.ones([3, 3])
    m, n = img.shape
    mask = mask / 9

    avg_img = np.zeros([m, n],dtype=np.uint8)
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
            
            avg_img[i, j]= temp
            
    cv.imshow("Original", img)
    cv.imshow("Average", avg_img)

def median_filter(image):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
    except:
        print("check image path!!")  
    m, n = img.shape
    
    med_img = np.zeros([m, n], dtype=np.uint8)
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
            med_img[i, j]= sorted(temp)[4]

    cv.imshow("Original", img)
    cv.imshow('Median_filtered', med_img)

def maximun_filter(image):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
    except:
        print("check image path!!")  
    m, n = img.shape
    max_img = np.zeros([m, n], dtype=np.uint8)
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
            max_img[i, j]= max(temp)

    cv.imshow("Original", img)
    cv.imshow('Maximun_filtered', max_img)
    

def minimun_filter(image):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
    except:
        print("check image path!!")  
    m, n = img.shape
    
    min_img = np.zeros([m, n], dtype=np.uint8)
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
            min_img[i, j]= min(temp)

    cv.imshow("Original", img)
    cv.imshow('Minimun_filtered', min_img)

def midpoint_filter(image):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
    except:
        print("check image path!!")  
    m, n = img.shape
    
    mid_img = np.zeros([m, n], dtype=np.uint8)
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
            mid_img[i, j]= (np.array(min(temp), dtype=float)+ max(temp))/2

    cv.imshow("Original", img)
    cv.imshow('Midpoint_filtered', mid_img)
    
def alpha_trimmed_filter(image,a=1):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
    except:
        print("check image path!!")  
    m, n = img.shape
    
    alp_img = np.zeros([m, n], dtype=np.uint8)
    
    # a= 2
    # ma = np.array([2, 0,1,4,9,3, 4, 5, 6, 7, 8, 19])
    # print(ma)
    # mas = sorted(ma)
    # print('sor: ',mas[a:-a])
    # print((sum(mas[a:-a]))/len(mas[a:-a]))
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
            temp_sorted= sorted(temp)
            alp_img[i, j]= sum(temp_sorted[a:-a])/len(temp_sorted[a:-a])

    cv.imshow("Original", img)
    cv.imshow('Alpha_trimmed_filtered', alp_img)
    
def main():
    #avg_filter("noise.jpg")
    #median_filter("noise.jpg")
    #maximun_filter("noise.jpg")
    #minimun_filter("noise.jpg")
    #midpoint_filter("noise.jpg")
    alpha_trimmed_filter("noise.jpg")



if __name__=="__main__":
    main()
    cv.waitKey(0)
