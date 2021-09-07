import cv2
import numpy as np

img = cv2.imread('pic2.png', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
num_labels, labels_im = cv2.connectedComponents(img)

def imshow_components(labels):

    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

imshow_components(labels_im)
