import cv2
import numpy as np
from matplotlib import pyplot as plt

def FFT(image, noise_type='diagonal'):
    img_noise = cv2.imread(image,0)
    f_noise = np.fft.fft2(img_noise)

    fshift_noise = np.fft.fftshift(f_noise)
    magnitude_spectrum_noise = 20*np.log(np.abs(fshift_noise))

    #filter position: centre
    #rows, cols = img_noise.shape
    #crow,ccol = int(rows/2) , int(cols/2)
    if noise_type=='diagonal':
        mask_position = [(65,74),(106,97),(23,55), (46,67)]
        for i in mask_position:
            fshift_noise[i[0]-8:i[0]+8, i[1]-8:i[1]+8]=1 #applying the mask diagonal
    else:
        mask_position = [(44,87),(9,87),(126,87), (167,87)]
        for i in mask_position:
            fshift_noise[i[0]-8:i[0]+8, i[1]-8:i[1]+8]=1 #applying the mask vertical

    magnitude_spectrum_filter = 20*np.log(np.abs(fshift_noise))

    # shift back (we shifted the center before)
    f_ishift = np.fft.ifftshift(fshift_noise)

    # inverse fft to get the image back 
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    plot_results(img_noise,magnitude_spectrum_noise,magnitude_spectrum_filter,img_back)


def plot_results(img_noise,magnitude_spectrum_noise,magnitude_spectrum_filter,img_back):
    titles = ['Noise Image', 'Magnitude Spectrum noise', 'Image after HPF', 'Final Result']
    images = [img_noise,magnitude_spectrum_noise,magnitude_spectrum_filter,img_back]
    for i in range(1,5):
        plt.subplot(220+i),plt.imshow(images[i-1], cmap = 'gray')
        plt.title(titles[i-1]), plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    FFT('eagle_noise_horizontal.png',noise_type='horizontal')

if __name__=="__main__":
    main()
    #cv.waitKey(0)

"""
# plt.subplot(221),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(221),plt.imshow(img_noise, cmap = 'gray')
plt.title('Noise Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum_noise, cmap = 'gray')
plt.title('Magnitude Spectrum noise'), plt.xticks([]), plt.yticks([])
plt.show()    
"""