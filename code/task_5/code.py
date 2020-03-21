#! python

import cv2
import numpy as np
import os
from glob import iglob
import json
import math
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 



class planarHomography():
    def __init__(self,path):
        # termination criteria 
        self.path = path

    def loadSingleImage(self,image_name):
        img = cv2.imread(self.path+"/"+image_name)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Single Loaded Image to check",gray)
        cv2.waitKey()
        return gray
    
if __name__ == "__main__":
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    """
        This might not be the ideal world frame and you may want to define the world origin and
        the unit length based on the field of view of the camera and metric distance of real-world
        objects, especially if you want to show the reconstruction like that in Figure 3 (left). 
        Hence, you may need to translate and scale the coordinates of these points. In this example 
        as shown in Figure 3 (left), the 2D points on the world plane used in calculating the homography 
        are scaled by 10 translated to (300, 800, 0). This means the length of each cell of the chessboard 
        pattern is 10 unit length, and hence, occupies 10 pixels. The top-left corner of the chessboard is 
        at (300, 800) on the warped image. The camera pose in 3D is recovered using PnP as explained in the next task.
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    pHomography = planarHomography("../../images/task_5")

    ### Step 1 ###
    ### Load the image and the camera parameters
    gray = pHomography.loadSingleImage("left_0.png")
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(gray, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey()

    cv2.destroyAllWindows()

    ### Step 2 ###
    ### Undistort the loaded image and extract 2D-2D point corrospondence

    ### Step 3 ###
    """ 
    Calculate the planar homography
    Reference Notes: https://www.learnopencv.com/homography-examples-using-opencv-python-c/
                     https://drive.google.com/file/d/1yHtHPP26Q7N32MJaXJihSGRrPMT7pf7-/view
    """

    ### Wrap the image ###
    """
        Reference function given in the Class note is "warpPerspective"
    
    """