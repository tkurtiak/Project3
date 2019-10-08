import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 


# Save/Load File name for RGB values
filename = 'MaroonWindow.npy'
# Selected training img
imgname = 'Images/InitislSet.png'
# load image
img = cv2.imread(imgname)
img_thresh = cv2.imread(imgname)
# load current training dataset
BGR_set = np.load(filename)

covariance = np.cov(BGR_set)
icovariance = np.linalg.inv(covariance)
mean = np.mean(BGR_set,axis = 1)

prior = 1/2 #2 options, either it is the color or it is not the color


#p = 1/((2*3.14159)^3*np.norm(covariance))*np.exp(.5*(X-mean).T*icovariance*(X-mean))
p0 = 1/(((2*3.14159)**3*np.linalg.norm(covariance)))**(0.5)

# Loop through pixels and compare each value to the threshold.
x=0
y=0

for col in img:
	for pixel in col:
		X = pixel
		p = p0*np.exp(.5*(X-mean).T*icovariance*(X-mean))
		post = np.linalg.norm((p*prior)/prior)
		if post >= 0.95:
			# set the pixel to yellow to stand out
			img_thresh[y,x] = [0,255,255]
		x = x+1
	x = 0
	y = y+1

plt.subplot(2,1,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(2,1,2),plt.imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
plt.show()