import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#img=train_images[0,:,:]


def elastic_transform(image, alpha, sigma, random_state=None):
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    #print(indices)

    return map_coordinates(image, indices, order=1).reshape(shape)

'''
plt.figure(figsize=[10, 5])
plt.subplot(141)
img1=elastic_transform(img,10,0.1)
plt.imshow(img1, cmap='gray')
plt.title("σ = 0.1")
plt.axis('off')
plt.subplot(142)
img2=elastic_transform(img,8,2)
plt.imshow(img2, cmap='gray')
plt.title("σ = 2")
plt.axis('off')
plt.subplot(143)
img3=elastic_transform(img,8,4)
plt.imshow(img3, cmap='gray')
plt.title("σ = 4")
plt.axis('off')
plt.subplot(144)
img4=elastic_transform(img,8,20)
plt.imshow(img4, cmap='gray')
plt.title("σ = 20")
plt.axis('off')
#plt.subplot(122)
#plt.imshow(elastic_transform(train_images[5,:,:],30,3), cmap='gray')
#plt.title("Деформирана слика")
plt.show()
'''
