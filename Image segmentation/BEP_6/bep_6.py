import cv2 as cv
import numpy as np

def flood_recursive(image):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
        (thresh, img_bin) = cv.threshold(img, 20, 255, cv.THRESH_BINARY)
    except:
        print("check image path!!")
    mask = np.zeros(np.asarray(numeric_matrix.shape)+2, dtype=np.uint8)
    start_pt = (y,x)
    if matrix_np[start_pt]:
        cv.floodFill(numeric_matrix, mask, start_pt, 255, flags=4)
    mask = mask[1:-1, 1:-1]
    matrix_np[mask==1] = "c"


def two_pass(image):
    try:
        img = cv.imread(image,0) # Read Img in BGR Colour
        (thresh, img_bin) = cv.threshold(img, 20, 255, cv.THRESH_BINARY)
    except:
        print("check image path!!")
    
    num_labels, labels = cv.connectedComponents(img_bin)
    
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    #print(np.unique(labels))
    #print(np.max(label_hue))
    blank_ch = 255*np.ones_like(label_hue) #same characteristic as label_hue
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    cv.imshow("Labeled Hueaaaaaaa", labeled_img)
    labeled_img[label_hue==0] = 0 #cambiando el fondo rojo a nego
    
    cv.imshow("Binarized", img_bin)
    cv.imshow("Labeled Hue", label_hue)
    cv.imshow("Labeled", labeled_img)

    # set bg label to black
    
def main():
    flood_recursive("shape.jpg")
    #two_pass("shape.jpg")

if __name__=="__main__":
    main()
    cv.waitKey(0)

#https://iq.opengenus.org/connected-component-labeling/
