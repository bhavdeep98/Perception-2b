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
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)
        # self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 25, 1e-5)
        self.left_imgs = []
        self.right_imgs = []
        self.objpoints = [] # 3D points in the real world space
        self.CameraParameters = namedtuple('CameraParameters', ['M', 'distCoeff', 'R', 't'])
        self.h = 0
        self.w = 0
        self.curruntImage = ""

    ### Load Images
    def load_single(self,image_name):
        self.curruntImage = image_name
        img = cv2.imread(self.path+"/"+image_name)
        cv2.imshow("Single Loaded Image to check",img)
        cv2.waitKey(500)
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
                cv2.waitKey(500)
        self.objpoints.append(objp)
        return corners2


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
        # print("ROI :",x, y, w, h)
        # print(dst)
        cv2.imwrite('calibresult'+fname+'.png', dst)
        return mapx,mapy

    def findHomography(self, img1, img2):
        # define constants
        MIN_MATCH_COUNT = 10
        MIN_DIST_THRESHOLD = 0.7
        RANSAC_REPROJ_THRESHOLD = 5.0

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # find matches
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < MIN_DIST_THRESHOLD * n.distance:
                good.append(m)


        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
            return H

        else: raise Exception("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

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

    ### Step 1 ###
    ### Load the image and the camera parameters
    gray = pHomography.load_single("left_0.png")

    call = []
    carr = []
    ## Load the files saved in the previous tasks
    call.append(np.loadtxt("../../parameters/left_provided/cameraMatrix.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    call.append(np.loadtxt("../../parameters/left_provided/cameraDistortion.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    carr.append(np.loadtxt("../../parameters/right_provided/cameraMatrix.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    carr.append(np.loadtxt("../../parameters/right_provided/cameraDistortion.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    
    # objs, imgpoints = cv2.findChessboardCorners(gray, (9, 6), None)
    corners = pHomography.find_corners([gray])
    # print(corners)
    ### Step 2 ###
    ### Undistort the loaded image and extract 2D-2D point corrospondence
    print("At Step 2")
    
    # outputImage_camera1 = cv2.undistortPoints(np.concatenate(imgpoints,axis=0),call[0],call[1])
    mapx,mapy = pHomography.undistort(gray,call,"left_undistorted")
    # print(os.getcwd())
    outputImage_camera = cv2.imread(r"calibresultleft_undistorted.png")
    cv2.imshow("Undistorted Image",outputImage_camera)
    cv2.waitKey(500)

    # print(gray)
    # print("imgpoints")
    # print(imgpoints)
    # print(np.array(imgpoints).reshape(9,6))
    # rec_corners = cv2.remap(corners, mapx, mapy, cv2.INTER_LINEAR)
    # print(rec_corners)
    h, w = outputImage_camera.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(call[0], call[1], (w, h), 1, (w, h))
    
    # corners_updated=np.expand_dims(corners,axis=1)
    temp = corners.reshape((-1,2))
    # print(temp.shape)
    # z = np.zeros((9*6,1), np.float32)
    # corners_updated = np.append(temp,z,1)
    # print(corners_updated.shape)
    # print(temp)
    print(newcameramtx)
    print(call[0])
    print(call[1])
    rec_corners = cv2.undistort(temp,call[0],call[1],None,newcameramtx)
    # undistort
    # mapx, mapy = cv2.initUndistortRectifyMap(call[0], call[1], None, newcameramtx, (w, h), 5)
    # rec_corners = cv2.remap(corners_updated, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    print(roi)
    # rec_corners = rec_corners[y:y + h, x:x + w]
    # print(rec_corners)
    
    # objs, imgpoints = pHomography.find_corners([outputImage_camera])
    # print("rec_imgpoints ***************")
    # print(rec_imgpoints)
    # rec_imgpoints = cv2.findChessboardCorners(outputImage_camera, (9, 6), None)
    # print(rec_imgpoints)
    objp_2 = np.zeros((6*9,3), np.float32)
    objp_2[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp_2 = objp_2*10
    objp_2[:,0] = objp_2[:,0] + 300
    objp_2[:,1] = objp_2[:,1] + 800
    print(objp_2.shape)
    
    ### Step 3 ###
    """ 
    Calculate the planar homography
    Reference Notes: https://www.learnopencv.com/homography-examples-using-opencv-python-c/
                     https://drive.google.com/file/d/1yHtHPP26Q7N32MJaXJihSGRrPMT7pf7-/view
    """
    h, status = cv2.findHomography(temp,objp_2)

    ### Wrap the image ###
    """
        Reference function given in the Class note is "warpPerspective"
    """

    im_out = cv2.warpPerspective(gray, h, (outputImage_camera.shape[1],outputImage_camera.shape[0]))
    cv2.imshow("Warped Source Image", im_out)
    cv2.waitKey()