import matplotlib.pyplot as plt
import numpy as np
import cv2

#then we have to specify translation matrix
"""
transition matrix :
[1,0,tx]
[0,1,ty]
#the first row of matrix is [1,0,tx] where tx is the number of pixels we shift the image left or right
#positive values of tx will shift to right
#negative values will shift to left
#the second row of matrix is [0,1,ty] where ty i the number of pixels we shift the image up or down
#positive values of ty will shift to down and negative to up
"""

test_file = './mnist_test.csv'
testSet = np.loadtxt(test_file, delimiter=',')
test_data = testSet[:, 1:785]
test_images = test_data.reshape(test_data.shape[0], 28, 28)
img=test_images[34, :, :]

plt.figure()
plt.subplot()
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


right = np.float32([[1,0,6],[0,1,0]])
left = np.float32([[1,0,-6],[0,1,0]])
leftUp = np.float32([[1,0,-6],[0,1,4]])
leftDown = np.float32([[1,0,6],[0,1,-4]])

#using above translation matrix , we shift the image to 20 pixel right and 40 pixel to down
img1 = cv2.warpAffine(img,right,(img.shape[1],img.shape[0]))
img2 = cv2.warpAffine(img,left,(img.shape[1],img.shape[0]))
img3 = cv2.warpAffine(img,leftUp,(img.shape[1],img.shape[0]))
img4 = cv2.warpAffine(img,leftDown,(img.shape[1],img.shape[0]))
#re=img1.reshape(1,784)
#print(img1.shape)
#print(re.shape)


plt.figure(figsize=[7, 5])
plt.subplot(221)
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.subplot(222)
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.subplot(223)
plt.imshow(img3, cmap='gray')
plt.axis('off')
plt.subplot(224)
plt.imshow(img4, cmap='gray')
plt.axis('off')



plt.show()
