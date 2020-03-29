import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from glob import iglob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Step (1): Load the image sequence and camera parameters. 
images = []
left_images = "../../images/task_6/left_*"
left_file_list = [f for f in iglob(left_images, recursive=True) if os.path.isfile(f)]
left_imgs = [cv2.imread(x) for x in left_file_list]
ite = 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=10, azim=0)
plt.title('3D point cloud: Use pan axes button below to inspect')

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
	plt.savefig("../../output/task_6/left_"+ str(left_file_list[ite].split('.png')[0].split('left_')[1]) + ".png")


	#Step (3): Estimate the camera pose using PNP

	object_points = np.float32([[0., 0., 0.], 
							 [1., 0., 0.],
							 [1., 1., 0.], 
							 [0., 1., 0.]])

	# convert format from list to np
	corners = np.asarray(corners).squeeze()

	( _, rvec, tvec) = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

	R = np.matrix(cv2.Rodrigues(rvec)[0]).T

	t =(-R * tvec)*5

	print("\n********************" + left_file_list[ite]+ "********************\n")
	ite+=1

	print("R:")
	print(R)

	print("t:")
	print(t)

	# Step (4): Check the camera pose estimation results

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

	cam_frustrum3 = np.squeeze(np.asarray(cam_frustrum2))

	#ax.plot(cam_frustrum1[:,0], cam_frustrum1[:,1], cam_frustrum1[:,2])
	ax.plot(cam_frustrum3[:,0], cam_frustrum3[:,1], cam_frustrum3[:,2])
	# plot top left corner as red dot
	ax.scatter(cam_frustrum3[:,0][0], cam_frustrum3[:,1][0], cam_frustrum3[:,2][0], c='r', marker='o',zdir="z")

# Plot square as representation of ArUco marker
cam_aruco = np.array([[0,0,0],
	           [10,0,0],
	           [10,10,0],
	           [0,10,0],
	           [0,0,0]])

ax.plot(cam_aruco[:,0], cam_aruco[:,1], cam_aruco[:,2])
ax.scatter(0,0,0, c='r', marker='o',zdir="z")

#plt.show()

ax.view_init(elev=-135, azim=-45)
fig.savefig("../../output/task_6/view1.png")
ax.view_init(elev=-140, azim=-10)
fig.savefig("../../output/task_6/view2.png")
ax.view_init(elev=-90, azim=0)
fig.savefig("../../output/task_6/view3.png")
ax.view_init(elev=-145, azim=-5)
fig.savefig("../../output/task_6/view4.png")