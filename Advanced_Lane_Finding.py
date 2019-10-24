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

# line class decleration
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x value of the line
        self.current_x = False
        # last coefficients 
        self.recent_coeff = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in meters
        self.radius_of_curvature = None 
        # recent optimal radius value over last n frames
        self.recent_radius = None 
        #distance in meters of vehicle center from the line
        self.offset = None 
        # recent optimal distance value over last n frames
        self.recent_offset = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

        self.fitX = None
