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
        self.path = path
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        self.left_imgs = []
        self.right_imgs = []
        self.objpoints = [] # 3D points in the real world space
        self.CameraParameters = namedtuple('CameraParameters', ['M', 'distCoeff', 'R', 't'])
        self.h = 0
        self.w = 0

    ### Load Images
    def load_single(self,image_name):
        img = cv2.imread(self.path+"/"+image_name)
        cv2.imshow("Single Loaded Image to check",img)
        cv2.waitKey()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        self.h = h
        self.w = w
        return gray

    def load_images(self):
        left_images = self.path+"/left*"
        left_file_list = [f for f in iglob(left_images, recursive=True) if os.path.isfile(f)]

        right_images = self.path+"/right*"
        right_file_list = [f for f in iglob(right_images, recursive=True) if os.path.isfile(f)]

        self.left_imgs = [cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2GRAY) for x in left_file_list]
        self.right_imgs = [cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2GRAY) for x in right_file_list]

        return  self.left_imgs,self.right_imgs

    def find_corners(self,imgs):

        objp = np.zeros((9 * 6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objs = []
        imgpoints = []
        for x in imgs:
            ret, corners = cv2.findChessboardCorners(x, (9, 6), None)
            if ret==True:
                objs.append(objp)
                corners2 = cv2.cornerSubPix(x, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(x, (9, 6),
                                                  corners, ret)
                cv2.imshow("image", x)
                cv2.waitKey()
        self.objpoints.append(objp)
        return objs,imgpoints


    def calibratecamera(self,imagepoints):
        # print(self.objpoints)
        calib = cv2.calibrateCamera(self.objpoints,imagepoints,(9,6),None,None)
        return calib

    # Calib : ret, mtx, dist, rvecs, tvecs
    def undistort(self,img, calib,fname):
        # get image camera matrix
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(calib[0], calib[1], (w, h), 1, (w, h))

        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(calib[0], calib[1], None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite('calibresult'+fname+'.png', dst)

        return mapx,mapy

    def stereo_calibrate(self,imgpoints_l,imgpoints_r,call,carr):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.objpoints, imgpoints_l, 
                                imgpoints_r, call[0], call[1], carr[0], 
                                carr[1], (self.w, self.h), 
                                R=None, T=None, E=None, F=None,
                                criteria=self.stereocalib_criteria,
                                flags=cv2.CALIB_FIX_INTRINSIC)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        print('')

        camera_model = dict([('ret',ret),('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        return camera_model

    
if __name__ == "__main__":
    
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
    pHomography_2 = planarHomography("../../images/task_5")
    

    ### Step 1 ###
    ### Load the image and the camera parameters
    gray = pHomography.load_single("left_0.png")
    gray_2 = pHomography_2.load_single("right_0.png")
    
    # l,r = pHomography.load_images()
    # lobjs, limgpoints = pHomography.find_corners(l)
    # robjs, rimgpoints = pHomography.find_corners(r)
    # print(limgpoints)
    # call= pHomography.calibratecamera(limgpoints)
    # carr= pHomography.calibratecamera(rimgpoints)
    call = []
    carr = []
    ## Load the files saved in the previous tasks
    call.append(np.loadtxt("../../parameters/left/cameraMatrix.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    call.append(np.loadtxt("../../parameters/left/cameraDistortion.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    carr.append(np.loadtxt("../../parameters/right/cameraMatrix.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    carr.append(np.loadtxt("../../parameters/right/cameraDistortion.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    
    # objs, imgpoints = cv2.findChessboardCorners(gray, (9, 6), None)
    objs, imgpoints = pHomography.find_corners([gray])
    objs_2, imgpoints_2 = pHomography_2.find_corners([gray_2])

    # print("imgpoints_2 --------")
    # print(imgpoints_2)
    ### Step 2 ###
    ### Undistort the loaded image and extract 2D-2D point corrospondence
    print("At Step 2")
    # stereo_camera_model = pHomography.stereo_calibrate(limgpoints,rimgpoints,call,carr)

    # outputImage_camera1 = cv2.undistortPoints(np.concatenate(imgpoints,axis=0),call[0],call[1])
    mapx,mapy = pHomography.undistort(gray,call,"left_undistorted")
    # print(os.getcwd())
    outputImage_camera = cv2.imread(r"calibresultleft_undistorted.png")
    cv2.imshow("Undistorted Image",outputImage_camera)
    cv2.waitKey()
    
    # print("imgpoints")
    # print(np.array(imgpoints))
    rec_imgpoints = cv2.remap(np.array(imgpoints), mapx, mapy, cv2.INTER_LINEAR)
    # objs, imgpoints = pHomography.find_corners([outputImage_camera])
    print("rec_imgpoints ***************")
    print(rec_imgpoints)

    #region Second Image for homography
    # gray_2 = pHomography_2.load_single("left_1.png")
    # objs_2, imgpoints_2 = pHomography_2.find_corners([gray_2])
    # print("imgpoints_2++++++")
    # print(np.array(imgpoints_2))
    
    mapx,mapy = pHomography_2.undistort(gray_2,carr,"right_undistorted")
    outputImage_camera_2 = cv2.imread(r"calibresultright_undistorted.png")
    cv2.imshow("Undistorted Image 2",outputImage_camera_2)
    cv2.waitKey()
    
    rec_imgpoints_2 = cv2.remap(np.array(imgpoints_2), mapx, mapy, cv2.INTER_LINEAR)
    print("rec_imgpoints_2")
    print(rec_imgpoints_2)
    # objs_2, imgpoints_2 = pHomography.find_corners([outputImage_camera_2])

    #endregion
    print(len(imgpoints))
    h, status = cv2.findHomography(np.array(imgpoints),np.array(imgpoints_2))

    im_out = cv2.warpPerspective(gray, h, (gray_2.shape[1],gray_2.shape[0]))

    cv2.imshow("Warped Source Image", im_out)

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