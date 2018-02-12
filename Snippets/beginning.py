import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage.color import rgb2gray
from skimage import data, measure, segmentation, filters, feature, morphology
from skimage.filters import gaussian

# bilateral filter
# active contore
# edge detection and transformation
from skimage.segmentation import active_contour

img = ndimage.imread("test_images/test4.jpg")

blur = cv2.bilateralFilter(img, 20, 100, 200)

plt.imshow(blur, "gray")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(thresh, im_bw) = cv2.threshold(imgray, 90, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
med = cv2.medianBlur(im_bw, 9)

plt.imshow(med, "gray", interpolation='nearest')
plt.figure()
plt.imshow(img, "gray", interpolation='nearest')
# Find contours at a constant value of 0.8
contours = measure.find_contours(med, 0.5, fully_connected="high", positive_orientation="low")

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
#
# s = np.linspace(0, 2*np.pi, 400)
# x = 220 + 100*np.cos(s)
# y = 100 + 100*np.sin(s)

#

#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
# ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()


# # Construct some test data
# x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
# r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
#
#
# plt.show()
