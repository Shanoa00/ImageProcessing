import cv2 as cv
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(15,15))

def plot_image(image,pos, title, tipo=None):
    ax=fig.add_subplot(2, 2, pos)
    ax.set_title(title)
    plt.axis('off')
    plt.imshow(image,cmap=tipo)

def read_image(image):
    imageb= cv.imread(image, 0)
    image=cv.imread(image)
    if image.shape[0]>600:
        image=cv.resize(image,(600,400))
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    threshMap = cv.threshold(imageb, 200, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    plot_image(image,1,'Original Image')
    plot_image(threshMap,2, 'Otsu', 'gray')
    return image

def static_Spectral_Saliency(image):
    image= read_image(image)
    #Static Spectral Saliency
    saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    threshMap = cv.threshold(saliencyMap, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    
    plot_image(saliencyMap,3,'Saliency Map')
    plot_image(threshMap,4,'Thresholded image', 'gray')
    plt.show()

def static_Saliency_FineGrained(image):
    image= read_image(image)
    #Static Spectral Saliency
    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    print(success)
    saliencyMap = (saliencyMap*255).astype("uint8")
    
    threshMap = cv.threshold(saliencyMap, 200, 250, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    
    plot_image(saliencyMap,3,'Saliency Map')
    plot_image(threshMap,4,'Thresholded image', 'gray')
    plt.show()

def main():
    #static_Spectral_Saliency("bird.jpg")
    static_Saliency_FineGrained("aa.jpg") #cricket, aa

if __name__=="__main__":
    main()
    cv.waitKey(0)

#https://www.cronj.com/blog/finding-region-of-interest-roi-saliency/