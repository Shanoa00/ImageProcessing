import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2    

"""
How to choose the correct wavelet????
*harr, db2, sym2 --> Want to find closely spaced features, choose wavelet with small support, because is small enought to separate the features of interest
symlet4, bior4.4 --> (orthogonal wavelets) for denoising imgs, (reduce small sparks in the image)
"""


def w2d(img, wavelet_type='db2', thres= .2):
    imArray = cv2.imread(img)
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs2 = pywt.dwt2(imArray, wavelet_type) #bior1.3
    cA, (cH, cV, cD) = coeffs2

    #filtering:
    print(cH.max(), cH.mean())
    #cH[cH>=thres]=0
    #cV[cV>=thres]=0
    #cD[cD>=thres]=0

    coeffs2= cA, (cH, cV, cD)

    # reconstruction
    reconstr=pywt.idwt2(coeffs2, wavelet_type);
    reconstr *= 255;
    reconstr =  np.uint8(reconstr)
    #Display result
    cv2.imshow('Original', imArray)
    cv2.imshow('image', reconstr)
    fig = plt.figure(figsize=(12, 3))
    titles = ['Approximation (cA)', ' Horizontal detail (cH)',
             'Vertical detail (cV)', 'Diagonal detail (cD)']
    for i, a in enumerate([cA, cH, cV, cD]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    #plt.show()

def main():
    w2d('lena.jpg', thres=.001)

if __name__=="__main__":
    main()
    cv2.waitKey(0)

# import numpy as np
# import matplotlib.pyplot as plt

# import pywt
# import pywt.data


# # Load image
# original = pywt.data.camera()

# # Wavelet transform of image, and plot approximation and details
# titles = ['Approximation (cA)', ' Horizontal detail (cH)',
#           'Vertical detail (cV)', 'Diagonal detail (cD)']
# coeffs2 = pywt.dwt2(original, 'db2') #bior1.3
# cA, (cH, cV, cD) = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([cA, cH, cV, cD]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()