# IMPORTS
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import glob
import helpers

# variable decleration
test_image = plt.imread("test_images/test4.jpg")
left_lane_line = Line()
right_lane_line = Line()
