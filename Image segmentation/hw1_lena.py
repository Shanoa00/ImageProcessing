from MyImageLib import *

# python version 3.7.9

# converting color to gray scale.
lena = Image()
lena.load("lena.jpg")
lena.printImgAttrib()
lena_gray = lena.toGrayImage()
lena_gray.printImgAttrib()
lena_gray.calcHist()


plotter = ImagePlotter()
plotter.imgpdfcdf(lena_gray)

# drawing a circle
cd = CircleDrawer()
clena = cd.drawCircle(lena)
plotter.image(clena)
plotter.show()

hmimg = HistogramManipulator()
binar = Binarization()
# sliding
lena_gray_slided = hmimg.slide(lena_gray, 100)
lena_gray_slided.calcHist()
plotter.imgpdfcdf(lena_gray_slided)

# stretching
lena_gray_stretched = hmimg.stretch(lena_gray)
lena_gray_stretched.calcHist()
plotter.imgpdfcdf(lena_gray_stretched)

# shrinking
lena_gray_shrinked = hmimg.shrink(lena_gray, [100, 150])
lena_gray_shrinked.calcHist()
plotter.imgpdfcdf(lena_gray_shrinked)

# stretching
lena_gray11 = binar.iterative(lena_gray)
#lena_gray11.calcHist()
plotter.imgpdfcdf(lena_gray11)



