# IMPORTS
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import glob
import helpers
import os
import math
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
        self.current_fit = [] 
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

class LineData():
    def __init__(self, allX, allY, currentX, fitx, recentFit, currentFit, recentR, currentR):
        #x values for detected line pixels
        self.allx = allX  
        #y values for detected line pixels
        self.ally = allY
        # x value of the line at the bottom of the image (= at the cars position)
        self.current_x = currentX
        # polynomial fit x values of the curvature
        self.fitX = fitx
        # last coefficients 
        self.recent_fit = recentFit
        #polynomial coefficients for the most recent fit
        self.current_fit = currentFit
        # recent optimal radius value over last n frames
        self.recent_radius = recentR
        #radius of curvature of the line in meters
        self.radius_of_curvature = currentR 
        
# variable decleration
test_image = plt.imread("test_images/test4.jpg")
left_lane_line = Line()
right_lane_line = Line()
new_calc_frames = 0
frame_fails = 0
curvature_tolerance = 300

# Camera calibration
op, ip = helpers.calibrateCamera()

def sanity_check(left_line_values, right_line_values):
    """
    This function checks the plausibility of a new calculated frame from prior data
    INPUT: new calculated values from left and right line
    OUTPUT: True,  if lines pass and are valid or 
            False, if lines does not pass a test and should be not considered
    """
    # curvature comparison to prior
    prior_lane_radius = helpers.get_lane_curvature(left_lane_line, right_lane_line)
    new_lane_radius = np.mean((left_line_values[2], right_line_values[2]))
    curvature_check = (prior_lane_radius * 3 < new_lane_radius) or (prior_lane_radius / 3 > new_lane_radius)
    # curvature parallelism
    left_radius = left_line_values[2]
    right_radius = right_line_values[2]
    curvature_parallelism = (left_radius * 2 < right_radius) or (right_radius * 2 < left_radius)
    # slope comparison
    left_slope = left_line_values[0][0]
    right_slope = right_line_values[0][0]
    slope_sign_check = (np.sign(left_slope) == np.sign(right_slope))
    slope_close_check = math.isclose(left_slope, right_slope, rel_tol=5e-1)
    slope_check = not slope_sign_check or not slope_close_check
 
    sanity_test = curvature_check or curvature_parallelism or slope_check 
    return not sanity_test

def new_calc_test(left_line_values, right_line_values):
    """
    This function checks the plausibility of a new calculated frame from histogram. This is not so strict as the sanity_check() function.
    INPUT: new calculated values from left and right line
    OUTPUT: True,  if lines pass and are valid or 
            False, if lines does not pass a test and should be not considered
    """
    # 6) Sanity checks
    # 6.1) curvature comparison to prior
    prior_lane_radius = helpers.get_lane_curvature(left_lane_line, right_lane_line)
    new_lane_radius = np.mean((left_line_values[2], right_line_values[2]))
    curvature_check = (prior_lane_radius * 5 < new_lane_radius) or (prior_lane_radius / 5 > new_lane_radius)
    # 6.2) curvature parallelism
    left_radius = left_line_values[2]
    right_radius = right_line_values[2]
    curvature_parallelism = (left_radius * 2 < right_radius) or (right_radius * 2 < left_radius)
    return not (curvature_parallelism or curvature_check)

def save_new_line_data(left_line_data : LineData, right_lane_data : LineData):
    """
    Function saves new Data to the two lines
    """
    left_lane_line.update_pixel_values(left_line_data.allx, left_line_data.ally, left_line_data.current_x, left_line_data.fitX)
    left_lane_line.set_recent_fit(left_line_data.recent_fit)
    left_lane_line.set_current_fit(left_line_data.current_fit)
    left_lane_line.set_recent_radius(left_line_data.recent_radius)
    left_lane_line.set_current_radius(left_line_data.radius_of_curvature)

    right_lane_line.update_pixel_values(right_lane_data.allx, right_lane_data.ally, right_lane_data.current_x, right_lane_data.fitX)
    right_lane_line.set_recent_fit(right_lane_data.recent_fit)
    right_lane_line.set_current_fit(right_lane_data.current_fit)
    right_lane_line.set_recent_radius(right_lane_data.recent_radius)
    right_lane_line.set_current_radius(right_lane_data.radius_of_curvature)

    left_lane_line.detected = True
    right_lane_line.detected = True

def process_frame(frame_image):
    """
    This function process a frame from a video. The result is calculated from histogram peaks or from prior data according to the sanity check.
    INPUT: RGB-image 
    OUTPUT: annotated RGB-image
    """
    global frame_fails
    # 1) Undistort
    undistort_img = helpers.undistortImage(frame_image, op, ip)
    # 2) Threshold
    thresholded_img = helpers.create_thresholded_binary_image(undistort_img)
    # 3) Transform
    transformed_img, M_inv = helpers.transform_road(thresholded_img)

    # Check if lines were calculated before
    if (left_lane_line.radius_of_curvature == None or right_lane_line.radius_of_curvature == None):
        # if not, calculate it by histogram peak analysis

        # 4) Identify lane pixels
        leftx, lefty, rightx, righty = helpers.find_pixels_by_histogram(transformed_img)
        # 5) Calculate curvature
        colored_transformed_img, left_line_values, right_line_values = helpers.calc_curvature(transformed_img, leftx, lefty, rightx, righty)
        
        # 6) Sanity check for first frame; select smaller radius, if check fails
        left_radius = left_line_values[2]
        right_radius = right_line_values[2]
        if (left_radius * 10 < right_radius):
            right_radius = left_radius
        elif (right_radius * 10 < left_radius):
            left_radius = right_radius
        
        # 7) Save lane line data
        save_new_line_data(
            LineData(leftx, lefty, left_line_values[3], left_line_values[1], left_line_values[0], left_line_values[0], left_radius, left_radius),
            LineData(rightx, righty, right_line_values[3], right_line_values[1], right_line_values[0], right_line_values[0], right_radius, right_radius)
            )

    elif (left_lane_line.detected == False or right_lane_line.detected == False) and frame_fails > 3:
        # if not and prior frames failed, calculate it by histogram peak analysis

        # 4) Identify lane pixels
        leftx, lefty, rightx, righty = helpers.find_pixels_by_histogram(transformed_img)
        # 5) Calculate curvature
        colored_transformed_img, left_line_values, right_line_values = helpers.calc_curvature(transformed_img, leftx, lefty, rightx, righty)

        # 6) Sanity check if new curvature makes sense
        if new_calc_test(left_line_values, right_line_values): 
            # For statistics
            global new_calc_frames
            new_calc_frames += 1

            # 7) Save lane line data
            save_new_line_data(
            LineData(leftx, lefty, left_line_values[3], left_line_values[1], left_line_values[0], left_line_values[0], left_line_values[2], left_line_values[2]),
            LineData(rightx, righty, right_line_values[3], right_line_values[1], right_line_values[0], right_line_values[0], right_line_values[2], right_line_values[2])
            )

            frame_fails = 0
        else:
            # if not, try next frame
            frame_fails += 1
        
    else:
        # If there is prior data, use it for next search

        # 4) Identify lane pixels
        leftx, lefty, rightx, righty = helpers.find_pixels_by_prior(transformed_img, left_lane_line.allx, left_lane_line.ally, right_lane_line.allx, right_lane_line.ally)
        # 5) Calculate curvature
        colored_transformed_img, left_line_values, right_line_values = helpers.calc_curvature(transformed_img, leftx, lefty, rightx, righty)
        # 6) Sanity checks
        if (not sanity_check(left_line_values, right_line_values)):
            left_lane_line.detected = False
            right_lane_line.detected = False
            frame_fails += 1
        else:
            # 7) Update lane line data
            save_new_line_data(
            LineData(leftx, lefty, left_line_values[3], left_line_values[1], left_lane_line.current_fit, left_line_values[0], np.mean((left_lane_line.radius_of_curvature, left_lane_line.recent_radius)), np.mean((left_line_values[2], left_lane_line.recent_radius))),
            LineData(rightx, righty, right_line_values[3], right_line_values[1], right_lane_line.current_fit, right_line_values[0], np.mean((right_lane_line.radius_of_curvature, right_lane_line.recent_radius)), np.mean((right_line_values[2], right_lane_line.recent_radius)))
            )

    # 8) Warp back results
    result_img = helpers.warp_back_results(colored_transformed_img, undistort_img, M_inv, left_lane_line, right_lane_line)
    return result_img

def test_video_with_prior():
    ## Video
    output = 'output_videos/project.mp4'
    clip = VideoFileClip("test_videos/project_video.mp4").subclip(15, 30)
    processed_clip = clip.fl_image(process_frame)
    processed_clip.write_videofile(output, audio=False)
    print("New calculated frames: " + str(new_calc_frames))

test_video_with_prior()