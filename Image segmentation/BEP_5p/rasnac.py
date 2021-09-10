import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def rasnac(image):
    img=cv.imread(image,0)
    ret, th1 = cv.threshold(img, 210, 145, cv.THRESH_BINARY) 
    #Apply a bluur flter just to focus in the main area and eliminate noise
    blur = cv.GaussianBlur(img, (7, 7), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 200, 250, apertureSize = 3)
    y, x = np.where(canny)
    plt.scatter(x, y)
    

    minLineLength = 1
    maxLineGap = 10
    #lines = cv.HoughLinesP(canny,1,np.pi/180,100,minLineLength,maxLineGap)
    lines = cv.HoughLinesP(canny, 1, np.pi/180, 30, maxLineGap=200)
    for x1,y1,x2,y2 in lines[0]:
        cv.line(canny,(x1,y1),(x2,y2),(255,255,255),2)
        plt.plot([x1,x2],[y1,y2], color = 'r', linewidth = '4.5')

    cv.imshow("canny",canny )
    cv.imshow("lines",img)
    plt.gca().invert_yaxis()
    plt.show()

def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """
 
    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero
 
    # find a line model for these points
    m = (points[1,1] - points[0,1]) / (points[1,0] - points[0,0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1,1] - m * points[1,0]                                     # y-intercept of the line
 
    return m, c

def find_intercept_point(m, c, x0, y0):
    """ find an intercept point of the line model with
        a normal from point (x0,y0) to it
    :param m slope of the line model
    :param c y-intercept of the line model
    :param x0 point's x coordinate
    :param y0 point's y coordinate
    :return intercept point
    """
 
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c
 
    return x, y

def ransac_plot(n, x, y, m, c, final=False, x_in=(), y_in=(), points=()):
    """ plot the current RANSAC step
    :param n      iteration
    :param points picked up points for modeling
    :param x      samples x
    :param y      samples y
    :param m      slope of the line model
    :param c      shift of the line model
    :param x_in   inliers x
    :param y_in   inliers y
    """
 
    fname = "output/figure_" + str(n) + ".png"
    line_width = 1.
    line_color = '#0080ff'
    title = 'iteration ' + str(n)
 
    if final:
        line_width = 3.
        line_color = 'red'
        title = 'Final Solution'
    
 
    plt.figure("Ransac", figsize=(15., 15.))
    
    # grid for the plot
    grid = [min(x) - 10, max(x) + 10, min(y) - 20, max(y) + 20]
    plt.axis(grid)
 
    # put grid on the plot
    #plt.grid(b=True, which='major', color='0.75', linestyle='--')
    #plt.xticks([i for i in range(min(x) - 10, max(x) + 10, 5)])
    #plt.yticks([i for i in range(min(y) - 20, max(y) + 20, 10)])
 
    # plot input points
    plt.plot(x[:,0], y[:,0], marker='o', label='Input points', color='black', linestyle='None', alpha=0.4)
 
    # draw the current model
    plt.plot(x, m*x + c, 'r', label='Line model', color=line_color, linewidth=line_width)
    
    # draw inliers
    if not final:
        plt.plot(x_in, y_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)
 
    # draw points picked up for the modeling
    if not final:
        plt.plot(points[:,0], points[:,1], marker='o', label='Picked points', color='#0000cc', linestyle='None', alpha=0.6)
 
    plt.title(title)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    

def rasnac2(image):
    img=cv.imread(image,0)
    ret, th1 = cv.threshold(img, 210, 145, cv.THRESH_BINARY) 
    #Apply a bluur flter just to focus in the main area and eliminate noise
    blur = cv.GaussianBlur(img, (7, 7), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 200, 250, apertureSize = 3)
    fy, fx = np.where(canny)
    #n_inputs = np.ones(fx.shape)
    #n_outputs = 1

    x= np.expand_dims(fx, axis=1)
    y= np.expand_dims(fy, axis=1)
    data = np.hstack( (x,y) )

    
    ratio = 0.
    model_m = 0.
    model_c = 0.

    # Ransac parameters
    ransac_iterations = 10  # number of iterations
    ransac_threshold = 1    # threshold
    ransac_ratio = 0.6      # ratio of inliers required to assert
                            # that a model fits well to data
    
    # perform RANSAC iterations
    for it in range(ransac_iterations):
    
        # pick up two random points
        n = 2
    
        all_indices = np.arange(x.shape[0])
        np.random.shuffle(all_indices)
    
        indices_1 = all_indices[:n]
        indices_2 = all_indices[n:]

        maybe_points = data[indices_1,:]
        test_points = data[indices_2,:]
    
        # find a line model for these points
        m, c = find_line_model(maybe_points)
    
        x_list = []
        y_list = []
        num = 0
    
        # find orthogonal lines to the model for all testing points
        for ind in range(test_points.shape[0]):
    
            x0 = test_points[ind,0]
            y0 = test_points[ind,1]
    
            # find an intercept point of the model with a normal from point (x0,y0)
            x1, y1 = find_intercept_point(m, c, x0, y0)
    
            # distance from point to the model
            dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
            # check whether it's an inlier or not
            if dist < ransac_threshold:
                x_list.append(x0)
                y_list.append(y0)
                num += 1
 
        # in case a new model is better - cache it
        if num/float(x.shape[0]) > ratio:
            ratio = num/float(x.shape[0])
            model_m = m
            model_c = c
    
        # we are done in case we have enough inliers
        if num > x.shape[0]*ransac_ratio:
            print ('The model is found !')
            break
    
    # print ('\nFinal model:\n')
    # print ('  ratio = ', ratio)
    # print ('  model_m = ', model_m)
    # print ('  model_c = ', model_c)

    #x, model_m*x + model_c
    yy=model_m*x + model_c
    print(yy.min())
    cv.imshow("canny",canny )
    cv.imshow("lines",img)
    
    # plot the final model
    ransac_plot(0, x,y, model_m, model_c, True)

def main(): 
    rasnac("road2.jpg") ##using library
    #rasnac2("road2.jpg")

if __name__=="__main__":
    main()
    cv.waitKey(0)

#https://salzis.wordpress.com/2014/06/10/robust-linear-model-estimation-using-ransac-python-implementation/
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html