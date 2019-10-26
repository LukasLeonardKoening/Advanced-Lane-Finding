# IMPORTS
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import glob
import helpers
import os
from moviepy.editor import VideoFileClip

# line class decleration
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x value of the line at the bottom of the image (= at the cars position)
        self.current_x = False
        # last coefficients 
        self.recent_coeff = []
        #polynomial coefficients for the most recent fit
        self.current_fit = []#[np.array([False])]  
        #radius of curvature of the line in meters
        self.radius_of_curvature = None 
        # recent optimal radius value over last n frames
        self.recent_radius = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # polynomial fit x values of the curvature
        self.fitX = None
    
    def update_pixel_values(self, n_allx, n_ally, n_currentx, n_fitx):
        self.allx = n_allx
        self.ally = n_ally
        self.current_x = n_currentx
        self.fitX = n_fitx

    def set_recent_fit(self, n_recent_coeff):
        self.recent_coeff = n_recent_coeff

    def set_current_fit(self, n_current_fit):
        self.current_fit = n_current_fit

    def set_current_radius(self, n_radius):
        self.radius_of_curvature = n_radius
    
    def set_recent_radius(self, n_recent_r):
        self.recent_radius = n_recent_r

# variable decleration
test_image = plt.imread("test_images/test4.jpg")
left_lane_line = Line()
right_lane_line = Line()

# Camera calibration
op, ip = helpers.calibrateCamera()

def test_from_scratch(test_image):
    undistort_img = helpers.undistortImage(test_image, op, ip)

    thresholded_img = helpers.create_thresholded_binary_image(undistort_img)

    transformed_img, M_inv = helpers.transform_road(thresholded_img)

    identified_lines_image, left_lane_line_1, right_lane_line_1 = helpers.calc_curvature(transformed_img, left_lane_line, right_lane_line)
    
    result = helpers.warp_back_results(identified_lines_image, undistort_img, M_inv, left_lane_line_1, right_lane_line_1)

    return result

# im = test_from_scratch(test_image)
# plt.imshow(im)
# plt.show()

def test_video_without_prior():
    ## Video
    output = 'output_videos/project.mp4'
    clip = VideoFileClip("test_videos/project_video.mp4").subclip(0,5)
    processed_clip = clip.fl_image(test_from_scratch)
    processed_clip.write_videofile(output, audio=False)

test_video_without_prior()