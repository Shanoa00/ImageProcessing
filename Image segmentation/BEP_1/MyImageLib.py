import cv2 as cv
import numpy as np
from matplotlib import image, pyplot as plt
from numpy.core.getlimits import _get_machar

# python version 3.7.9

class Image:

    def __init__(self) -> None:
        self.img = 0

    def setImg(self, img):
        self.img = np.uint8(img)
        self.levelMin = self.img.min()
        self.levelMax = self.img.max()

    def load(self, filename):
        self.setImg(cv.imread(filename, cv.IMREAD_COLOR))

    def printImgAttrib(self):
        print(self.img.dtype)
        print(self.img.shape)
        print(self.img.ndim)

    def toGrayImage(self):

        if not self.isGrayScale():
            img_f16 = np.float16(self.img)
            img_u8 = np.array(img_f16.mean(axis=2), dtype=np.uint8)
            # g.img = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
            g = GrayImage()
            g.setImg(img_u8)
            return g

        else:
            return self
        
    def isGrayScale(self):
        return self.img.ndim == 2


class GrayImage(Image):

    def __init__(self):
        super().__init__()
        self.pdf = 0
        self.cdf = 0
        self.levelMin = 0
        self.levelMax = 0

    def load(self, filename):
        self.setImg(cv.imread(filename, cv.IMREAD_GRAYSCALE))

    def calcHist(self):
        self.pdf = cv.calcHist([self.img], [0], None, [256], [0, 256])
        self.cdf = np.cumsum(self.pdf)

    def imageStats(self):
        print("image gray low  = {}".format(self.levelMin))
        print("image gray high = {}".format(self.levelMax))


class CircleDrawer():

    def __init__(self):
        # default parameters
        self.circleColor = (255, 124, 53)
        self.circleRadius = 20
        self.circleThickness = 10
        self.circleCenter = (120, 120)

    def setCircleColor(self, col):
        self.circleColor = col

    def setCircleThickness(self, thickness):
        self.circleThickness = thickness

    def setCircleRadius(self, radius):
        self.circleRadius = radius

    def drawCircle(self, imgObj):
        cImg = cv.circle(imgObj.img, self.circleCenter,
                             self.circleRadius, self.circleColor, self.circleThickness)
        circleObj = GrayImage()
        circleObj.setImg(cImg)
        return circleObj


class HistogramManipulator():

    def __init__(self):
        super().__init__()

    def slide(self, imgObj, delta):
        
        tmp = np.float16(imgObj.img) + delta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        
        g = GrayImage()
        g.setImg(tmp)
        
        return g

    def stretch(self, imgObj):
        
        tmp = self.__calcNormalizedImage(imgObj) * 255
        
        g = GrayImage()
        g.setImg(tmp)
        
        return g

    def shrink(self, imgObj, r):
        
        tmp = self.__calcNormalizedImage(imgObj) * (r[1] - r[0]) + r[0]
        
        g = GrayImage()
        g.setImg(tmp)
        
        return g

    def __calcNormalizedImage(self, imgObj):
        
        minAdj = np.float16(imgObj.img) - imgObj.levelMin
        scale = imgObj.levelMax - imgObj.levelMin
        
        return minAdj / scale

    def equalize(self, imgObj):
        
        img = cv.equalizeHist(imgObj.img)

        g = GrayImage()
        g.setImg(img)

        return g


class BinaryImage(GrayImage):
    
    def __init__(self):
        super().__init__()

    def isBinary(self):
        return True


class Binarization():

    def static(self, imgObj, thresh):
        
        # *255 scaling for display purpose
        tmp = (np.float16(imgObj.img) > thresh) * 255
        
        g = GrayImage()
        g.setImg(tmp)

        return g

    def iterative(self, imgObj, T0=127, maxIter=200):
        
        img = imgObj.img

        T = T0
        for i in range(0, maxIter):
            u1 = np.array(img[img <= T], dtype=np.float16).mean()
            u2 = np.array(img[img > T], dtype=np.float16).mean()
            Tnew = (u1 + u2) / 2
            
            if abs(Tnew - T) < 0.1:
                break
            else:
                T = Tnew
        
        return self.static(imgObj, T), T


    def otsu(self, imgObj):
 
        thresh, tmp = cv.threshold(imgObj.img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        
        g = BinaryImage()
        g.setImg(tmp)

        return g, thresh

    def myOtsu(self, imgObj):
        
        img = np.float64(imgObj.img)

        bwr = np.zeros((256,))
        
        for T in range(1, 255):
            
            C0 = img <= T
            w0 = sum(sum(C0)) / img.size
            u0 = np.array(img[C0]).mean()
            v0 = np.array(img[C0]).var()
            
            C1 = img > T
            w1 = 1 - w0
            u1 = np.array(img[C1]).mean()
            v1 = np.array(img[C1]).var()
            
            sigma_between = w0 * w1 * (u0 - u1) ** 2
            # sigma_within = w0 * v0 + w1 * v1

            bwr[T] = sigma_between
        
        thresh = np.argmax(bwr)
        
        return self.static(imgObj, thresh), thresh


class ImageFilter(GrayImage):

    def __init__(self) -> None:
        self.mask = 0
        self.N = 0
        self.img = 0
        self.size = (0, 0)

    def setImg(self, imgObj):
        self.img = imgObj.img
        self.size = imgObj.img.shap

    def setMask(self, mask):
        self.mask = mask
        self.N = np.int32(mask.shape[0] / 2)

    def getAverage(self, N):
        return np.ones((N, N), dtype=np.float64) / N ** 2

    def __findRepresentative(self, x, y):
        c = 0
        
        c = self.img[rng_x, rng_y] * self.mask[i+self.N, j+self.N]
        return c.sum(sum(c))

    def average(self, imgObj):
        g = GrayImage()
        g.setImg(cv.filter2D(np.float64(imgObj.img), -1, self.mask))
        return g

    def median(self, imgObj):
        pass

    def max(self, imgObj):
        pass

    def min(self, imgObj):
        pass

    def midpoint(self, imgObj):
        pass

    def alphaTrimmedMean(self, alpha):
        pass



class ImagePlotter:

    def __init__(self) -> None:
        self.winID = 0

    def image(self, imgObj):
        plt.imshow(imgObj.img, cmap="gray", vmin=0, vmax=255)

    def histpdf(self, imgObj):
        plt.plot(imgObj.pdf)

    def histcdf(self, imgObj):
        plt.plot(np.cumsum(imgObj.pdf))

    def imgpdfcdf(self, imgObj):
        f = plt.figure()
        f.add_subplot(1, 3, 1)
        self.image(imgObj)
        f.add_subplot(1, 3, 2)
        self.histpdf(imgObj)
        f.add_subplot(1, 3, 3)
        self.histcdf(imgObj)
        self.show()

    def show(self):
        plt.show()
