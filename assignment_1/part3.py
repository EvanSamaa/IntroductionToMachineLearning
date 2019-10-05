from matplotlib import pyplot
import numpy as np

img = pyplot.imread("pink_lake.png")

# add 0.25 to img
img_add = img + 0.25
img_add = np.clip(img_add,0,1)
pyplot.imsave("img_add.png", img_add)

# split channels of img
img_chan_0 = np.zeros(img.shape)
img_chan_0[:,:,0] = img[:,:,0]
img_chan_1 = np.zeros(img.shape)
img_chan_1[:,:,1] = img[:,:,1]
img_chan_2 = np.zeros(img.shape)
img_chan_2[:,:,2] = img[:,:,2]
pyplot.imsave("img_chan_0.png", img_chan_0)
pyplot.imsave("img_chan_1.png", img_chan_1)
pyplot.imsave("img_chan_2.png", img_chan_2)

# gray scale of img
img_gray = np.zeros(img.shape)
img_gray[:,:,0] = 0.299 * img_chan_0[:,:,0] + 0.587 * img_chan_1[:,:,1] + 0.114 * img_chan_2[:,:,2]
img_gray[:,:,1] = img_gray[:,:,0]
img_gray[:,:,2] = img_gray[:,:,0]
pyplot.imsave("img_gray.png", img_gray)

# img crop
img_crop = img[0:int(img.shape[0]/2), :, :]
pyplot.imsave("img_crop.png", img_crop)

# img flip
img_flip_vert = np.zeros(img.shape)
img_flip_vert = np.flip(img, 1)
pyplot.imsave("img_flip_vert.png", img_flip_vert)

