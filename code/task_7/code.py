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
import operator


class CV():
    def __init__(self,path):
        self.path = path
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25.4, 0.001)
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
        # cv2.imshow("Single Loaded Image to check",img)
        # cv2.waitKey()
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
        # print(h,w)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(calib[0], calib[1], (w, h), 1, (w, h))
        print(roi)
        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(calib[0], calib[1], None, newcameramtx, (w, h), 5)

        # print(mapx,mapy)
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


    def get_keypoints(self,kp_l,l1,i1,orb):
        keypoint_list_l = []
        for i, keypoint in enumerate(kp_l):
            #print("Keypoint:", i, keypoint)
            keypoint_list_l.append(keypoint)

        cmpfun = operator.attrgetter('response')
        keypoint_list_l.sort(key=cmpfun, reverse=True)

        distance = []
        radius_l = []
        keypoint_i = 0
        for keypoint in keypoint_list_l:
            # print("Keypoint:", keypoint.response)
            distance.append([])
            if keypoint_i == 0:
                distance[0].append(1)
            for index in range(keypoint_i):
                distance[keypoint_i].append(np.linalg.norm(np.array(keypoint.pt) - np.array(keypoint_list_l[index].pt)))
            radius_l.append(min(distance[keypoint_i]))
            # print(keypoint_i, " radius_l:", radius_l[keypoint_i])
            keypoint_i = keypoint_i + 1

        keypoint_list_l1 = np.c_[keypoint_list_l, radius_l]
        keypoint_list_l1 = sorted(keypoint_list_l1, key=lambda x:x[1], reverse=True)
        keypoint_list_l1 = np.delete(keypoint_list_l1, 1, axis=1).transpose()[0]

        img3_l = cv2.drawKeypoints(l1, keypoint_list_l1, None, color=(0,255,0), flags=0)
        plt.imsave("../../output/task_7/"+i1+"_suppressed_key_points.png", img3_l)
        keypoint_list_l1, des_l = orb.compute(l1, keypoint_list_l1)

        return keypoint_list_l1,keypoint_list_l,des_l

    def kps2pts(self,kps1, kps2, matches):
        pts1 = []
        pts2 = []
        dis = []
        for m in matches:
            pts2.append(kps2[m.trainIdx].pt)
            pts1.append(kps1[m.queryIdx].pt)
            dis.append(m.distance)
        return np.float32(pts1), np.float32(pts2), np.float32(dis)

    def task_7(self,i1,i2,call):
        l1 = self.load_single(i1)
        l2 = self.load_single(i2)

        l1x,l1y=self.undistort(l1,call,"l1")
        l2x,l2y=self.undistort(l2,call,"l2")

        orb = cv2.ORB_create()

        pref = i2.split(".")[0]+"_"+i1.split(".")[0]

        kp_l,des = orb.detectAndCompute(l1, None)
        img2_l = cv2.drawKeypoints(l1, kp_l, None, color=(0,255,0), flags=0)
        plt.imsave("../../output/task_7/"+i1+"_key_points.png", img2_l)
 
        kp_r,des = orb.detectAndCompute(l2, None)
        img2_r = cv2.drawKeypoints(l2, kp_r, None, color=(0,255,0), flags=0)
        plt.imsave("../../output/task_7/"+i2+"_key_points.png", img2_r)

        # left keypoints
        keypoint_list_l1,keypoint_list_l,des_l= self.get_keypoints(kp_l,l1,i1,orb)
        # right keypoints
        keypoint_list_l2,keypoint_list_r,des_r= self.get_keypoints(kp_r,l2,i2,orb)

        #matches
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_l, des_r)

        print(len(keypoint_list_l1),len(keypoint_list_l2),len(matches))
        img4 = cv2.drawMatches(l1, keypoint_list_l1, l2, keypoint_list_l2, matches, l1)
        plt.imsave("../../output/task_7/"+pref+"_matches.png", img4)#,plt.show()

        ptsl1,ptsl2,_ = self.kps2pts(keypoint_list_l1,keypoint_list_l2,matches)

        E, mask = cv2.findEssentialMat(np.float32(ptsl1), np.float32(ptsl2),cameraMatrix=call[0])

        ptsl1_E = ptsl1[mask.ravel() == 1]
        ptsl2_E = ptsl2[mask.ravel() == 1]
        matches_E = []
        for i in range(len(matches)):
            if mask[i] == 1:
                matches_E.append(matches[i]) 

        def filter(kpts,mask):
            ret = []
            for kp,m in zip(kpts,mask):
                if m[0]==1:
                    ret.append(kp)
            return ret
        # keypoint_list_l1 = filter(keypoint_list_l1,mask)
        # keypoint_list_l2 = filter(keypoint_list_l2,mask)
        print(len(keypoint_list_l1),len(keypoint_list_l2),len(matches_E))

        img5 = cv2.drawMatches(l1, keypoint_list_l1, l2, keypoint_list_l2, matches_E, l1)
        plt.imsave("../../output/task_7/"+pref+"_matches_E.png", img5)#,plt.show()

        points, R, t, mask = cv2.recoverPose(E, ptsl1_E, ptsl2_E, cameraMatrix=call[0])#, distanceThresh=10)


        ptsl1_E = ptsl1_E[mask.ravel() == 255]
        ptsl2_E = ptsl2_E[mask.ravel() == 255]
        matches_E2 = []
        for i in range(len(matches_E)):
            if mask[i] == 255:
                matches_E2.append(matches_E[i]) 

        t = t*10
        print(points,R,t,mask)

        M_r = np.hstack((R, t))
        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

        print("\n",M_r,"\n",M_l)

        P_l = np.dot(call[0],  M_l)
        P_r = np.dot(call[0],  M_r)

        #only inliers triangulated
        point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(ptsl1_E, axis=1), np.expand_dims(ptsl2_E, axis=1))
        # point_4d_hom = cv2.triangulatePoints(P_l, P_r, ptsl1, ptsl2)

        def make_homogeneous_pose(R, t):
            return np.concatenate((np.concatenate((R,t),axis=1),np.array([[0,0,0,1]])),axis=0)

        T = make_homogeneous_pose(R,t)
        cam_frustrum = np.array([[-1,-1,1,1],
                   [1,-1,1,1],
                   [1,1,1,1],
                   [-1,1,1,1],
                   [-1,-1,1,1],
                   [0,0,0,1],
                   [1,-1,1,1],
                   [1,1,1,1],
                   [0,0,0,1],
                   [-1,1,1,1]])

        cam_frustrum1 = cam_frustrum*4

        cam_frustrum2 = (T @ cam_frustrum.transpose()).transpose()*4

        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        point_3d=point_3d*10

        # print(point_3d)

        # plot with matplotlib
        Xs = [x[0] for x in point_3d]
        Ys = [x[1] for x in point_3d]
        Zs = [x[2] for x in point_3d]

        if max(Zs)> 500:
            point_3d=point_3d/10
            Xs = [x[0] for x in point_3d]
            Ys = [x[1] for x in point_3d]
            Zs = [x[2] for x in point_3d]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_xlim3d(-500,500)
        # ax.set_ylim3d(-500,500)
        # ax.set_zlim3d(-20,500)
        ax.view_init(elev=10, azim=0)
        ax.plot(cam_frustrum1[:,0], cam_frustrum1[:,1], cam_frustrum1[:,2])
        ax.plot(cam_frustrum2[:,0], cam_frustrum2[:,1], cam_frustrum2[:,2])
        ax.scatter(Xs, Ys, Zs, c='g', marker='o',zdir="z")

        plt.title('3D point cloud: Use pan axes button below to inspect')
        # plt.show()
        ax.view_init(elev=-135, azim=-45)
        plt.savefig("../../output/task_7/"+pref+"_cam_1.png")
        ax.view_init(elev=-140, azim=-10)
        plt.savefig("../../output/task_7/"+pref+"_cam_2.png")
        ax.view_init(elev=-90, azim=0)
        plt.savefig("../../output/task_7/"+pref+"_cam_3.png")
        ax.view_init(elev=-145, azim=-5)
        plt.savefig("../../output/task_7/"+pref+"_cam_4.png")





if __name__ == "__main__":


    cv = CV("../../images/task_7")

    call = []
    call.append(np.loadtxt("../../parameters/left/cameraMatrix.txt", delimiter=','))#, encoding='bytes', allow_pickle=True).item()
    call.append(np.loadtxt("../../parameters/left/cameraDistortion.txt", delimiter=','))

    # print(call)
    cv.task_7("left_0.png","left_1.png",call)
    cv.task_7("left_1.png","left_2.png",call)
    cv.task_7("left_2.png","left_3.png",call)
    cv.task_7("left_3.png","left_4.png",call)
    cv.task_7("left_4.png","left_5.png",call)
    cv.task_7("left_5.png","left_6.png",call)
    cv.task_7("left_6.png","left_7.png",call)
    cv.task_7("left_6.png","left_0.png",call)
    cv.task_7("left_6.png","left_1.png",call)
    cv.task_7("left_6.png","left_2.png",call)
    cv.task_7("left_5.png","left_1.png",call)
    cv.task_7("left_5.png","left_3.png",call)



    


    



 