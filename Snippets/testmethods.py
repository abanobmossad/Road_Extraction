from __future__ import print_function
from skimage.feature import multiblock_lbp, draw_multiblock_lbp
import numpy as np
from numpy.testing import assert_equal
from skimage.transform import integral_image
from matplotlib import pyplot as plt
# Create test matrix where first and fifth rectangles starting
# from top left clockwise have greater value than the central one.
test_img = np.zeros((9, 9), dtype='uint8')
test_img[3:6, 3:6] = 1
test_img[:3, :3] = 50
test_img[6:, 6:] = 50

print(test_img)
# First and fifth bits should be filled. This correct value will
#  be compared to the computed one.
correct_answer = 0b10001000

int_img = integral_image(test_img)
print(int_img)

lbp_code = multiblock_lbp(int_img, 0, 0, 3, 3)
print(lbp_code)
img = draw_multiblock_lbp(test_img, 0, 0, 90, 90,
                          lbp_code=lbp_code, alpha=0.5)


plt.imshow(img, interpolation='nearest')

plt.show()
