import  cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage import morphology
from scipy import ndimage
from skimage import feature

kernel = np.ones((5,5),np.uint8)
kernel_clos = np.ones((6,6),np.uint8)


imgray = cv2.imread('testing.png')

img = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)

plt.figure('img')
plt.imshow(img,'gray')


med = cv2.medianBlur(img,9)
plt.figure('med')
plt.imshow(med,'gray')

closing = cv2.morphologyEx(med, cv2.MORPH_CLOSE, kernel_clos)
plt.figure('closing')
plt.imshow(closing,'gray')

opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
plt.figure('opening')
plt.imshow(opening,'gray')


dst = ndimage.gaussian_filter(opening,3)

plt.figure('dst')
plt.imshow(dst,'gray')



blur = cv2.bilateralFilter(dst, 50, 100, 300)
plt.figure('blur')
plt.imshow(blur,'gray')





erosion = cv2.erode(blur,kernel,iterations = 1)

plt.figure('erosion')
plt.imshow(erosion,'gray')

(thresh, im_bw) = cv2.threshold(erosion, 2, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# edges2 = feature.canny(erosion, sigma=0)
#
plt.figure('im_bw')
plt.imshow(im_bw,'gray')

plt.show()


'''
#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 150

#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255


'''