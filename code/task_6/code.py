import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from glob import iglob
import os

#Step (1): Load the image sequence and camera parameters. 
images = []
left_images = "../../images/task_6/left_*"
left_file_list = [f for f in iglob(left_images, recursive=True) if os.path.isfile(f)]
left_imgs = [cv2.imread(x) for x in left_file_list]
ite = 0

for frame in left_imgs:

	camera_matrix = np.array([[423.27381306, 0, 341.34626532],
				   			[0, 421.27401756, 269.28542111],
				   			[0, 0, 1]])

	k1 = -0.43394157423038077
	k2 = 0.26707717557547866
	p1 = -0.00031144347020293427
	p2 = 0.0005638938101488364
	k3 = -0.10970452266148858

	dist_coeffs = np.array([k1, k2, p1, p2, k3])

	#plt.figure()
	#plt.imshow(frame)
	#plt.show()

	#Step (2): Detect the ArUco marker

	# Post processing
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

	# Results
	plt.figure()
	plt.imshow(frame_markers)
	for i in range(len(ids)):
	    c = corners[i][0]
	    plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
	plt.legend()
	plt.title(left_file_list[ite])
	#plt.show()
	plt.savefig("../../output/task_6/left_"+ str(ite) + ".png")


	#Step (3): Estimate the camera pose using PNP

	object_points = np.float32([[0., 0., 0.], 
							 [1., 0., 0.],
							 [1., 1., 0.], 
							 [0., 1., 0.]])

	# convert format from list to np
	corners = np.asarray(corners).squeeze()

	print("\n********************" + left_file_list[ite]+ "********************\n")
	ite+=1

	( _, rvec, tvec) = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

	rotM = cv2.Rodrigues(rvec)[0]
	rotM2 = np.matrix(rotM).T

	print("R:")
	print(rotM2)

	# PENDING: Need to figure it out the t vector
	"""
	cameraPosition = -np.matrix(rotM).T * np.matrix(tvec)
	print("cameraPosition")
	print(cameraPosition)

	print("t:")
	t = -rotM * tvec
	print(tvec)
	"""

	# PENDING: Step (4): Check the camera pose estimation results