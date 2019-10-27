# IMPORTS
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import glob
import helpers

# Meters per pixel
ym_per_pix = 22/720
xm_per_pix = 3.7/990

def calibrateCamera():
    """
    Calculates parameters needed for camera calibration with the help of the calibration images
    OUTPUT: Two arrays with object points (obj_points) and image points (img_points)
    """
    # Load images from folder camera_cal with the help of the glob library
    images = glob.glob("camera_cal/calibration*.jpg")

    # Arrays to save object~ / image points from all images
    obj_points = []
    img_points = []

    # Preperation of objectpoints 
    objp = np.zeros((6*9,3), np.float32)
    for y in range(6):
        for x in range(9):
            objp[9*y+x] = [x,y,0]

    # Loop over all calibration images
    for image_path in images:
        # Read in image
        image = mpl_img.imread(image_path)

        # Convert Image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        img = cv2.drawChessboardCorners(image, (9,6), corners, ret)

        # Append image and object points  to the arrays
        if ret:
            img_points.append(corners)
            obj_points.append(objp)

    return obj_points, img_points

def undistortImage(image, o_points, i_points):
    """
    Undistort Images from a camera
    INPUT: distorted image, object_points, image_points (both from calibrateCamera())
    OUTPUT: undistorted image or error on failure
    """
    # Convert to gray scale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate Camera matrix and distance coefficients 
    ret, cam_matrix, distance_coeff, rot_vec, trans_vec = cv2.calibrateCamera(o_points, i_points, gray.shape, None, None)
    if ret:
        # return undistorted image
        return cv2.undistort(image, cam_matrix, distance_coeff)
    else:
        # raise error if camera calibration fails
        raise ValueError("Can not undistort the given image!")

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Function calculates the directional gradient and applies given thresholds as binary mask
    INPUT:  img = RGB image
            orient = 'x' or 'y'
            sobel_kernel = Sobel kernel size, odd positive number
            threshold = Tuple of lower and upper threshold
    OUTPUT: binary image
    """
    # 1) Pick the value channel from HSV convertion
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    # 2) Take the derivative in given orientation = 'x' or 'y'
    sobel = cv2.Sobel(v, cv2.CV_64F, orient == 'x', orient == 'y')
    # 3) Take the absolute value of the derivative of gradient
    absSobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    uint8_sobel = np.uint8(255*absSobel/np.max(absSobel))
    # 5) Create a binary mask
    grad_binary = np.zeros_like(uint8_sobel)
    grad_binary[(uint8_sobel > thresh[0]) & (uint8_sobel < thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Function calculates the gradient magnitude and applies given thresholds as binary mask
    INPUT:  img = RGB image
            sobel_kernel = Sobel kernel size, odd positive number
            threshold = Tuple of lower and upper threshold
    OUTPUT: binary image
    """
    # 1) Pick the value channel from HSV convertion
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255 * magnitude / np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Function calculates the gradient direction and applies given thresholds as binary mask
    INPUT:  img = RGB image
            sobel_kernel = Sobel kernel size, odd positive number
            threshold = Tuple of lower and upper threshold
    OUTPUT: binary image
    """
    # 1) Pick the value channel from HSV convertion
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    # 2) Take the gradient in x and y separately
    xSobel = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    ySobel = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    x_abs = np.absolute(xSobel)
    y_abs = np.absolute(ySobel)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(y_abs, x_abs)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return dir_binary

def create_thresholded_binary_image(rgb_image):
    """
    Function takes in an RGB image and returns an thresholded binary image with the lane lines 
    """
    # Thresholds
    s_threshold = [100, 255]
    white_threshold = [200, 255]
    yellow_threshold = [14, 25]

    # Color conversions
    hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    s_hls = hls[:,:,2]

    # white and yellow mask
    h = hls[:,:,0]
    l = hls[:,:,1]
    white_mask = (l > white_threshold[0]) & (l <= white_threshold[1])
    yellow_mask = (h > yellow_threshold[0]) & (h <= yellow_threshold[1])

    # Color thresholds
    s_threshold = (s_hls > s_threshold[0]) & (s_hls <= s_threshold[1])
    color_based_threshold = s_threshold & (white_mask | yellow_mask)

    # Sobel thresholds
    ksize = 1 # Sobel kernel size

    gradx = abs_sobel_thresh(rgb_image, orient='x', sobel_kernel=ksize, thresh=(25, 100)) #20, 50
    grady = abs_sobel_thresh(rgb_image, orient='y', sobel_kernel=ksize, thresh=(25, 100)) #20, 50
    mag_binary = mag_thresh(rgb_image, sobel_kernel=ksize, mag_thresh=(35, 85))
    dir_binary = dir_threshold(rgb_image, sobel_kernel=ksize, thresh=(np.pi/4, 4*np.pi/10)) # np.pi/4, np.pi/3

    sobel_based_threshold = ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))

    # Binary image creation
    binary = np.zeros_like(s_hls)
    binary[(color_based_threshold | sobel_based_threshold)] = 1

    return binary

def transform_road(binary_image):
    """
    Function transforms an image into the 'bird view'-perspective
    INPUT: binary image from create_thresholded_binary_image()
    OUTPUT: transformed binary image and inversed matrix for undoing the transformation
    """
    # Image size
    xsize = binary_image.shape[1]
    ysize = binary_image.shape[0]

    # Select source points
    src = np.float32([[554, 465], [735, 465], [90, ysize], [xsize, ysize]])

    # Select destination points
    dest = np.float32([[0,0],[xsize,0],[0, ysize],[xsize,ysize]])

    # Perform transformation
    M = cv2.getPerspectiveTransform(src, dest)
    Minv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(binary_image, M, (xsize, ysize), flags=cv2.INTER_LINEAR)

    return warped, Minv

def find_pixels_by_histogram(trans_img):#, left_line, right_line):
    """
    Function finds pixels on the binary filtered image and selects pixels according to histogram peaks
    """
    # Image size
    xsize = trans_img.shape[1]
    ysize = trans_img.shape[0]

    # Create an output image
    #out_img = np.dstack((trans_img, trans_img, trans_img))

    # Calculate initial left and right x
    part_of_image = trans_img[int(trans_img.shape[0]//2):, :]
    histogram = np.sum(part_of_image, axis=0)

    middle = np.int(histogram.shape[0]/2)
    init_leftx = np.argmax(histogram[:middle])
    init_rightx = np.argmax(histogram[middle:]) + middle

    # Parameters
    n_window = 10
    window_height = np.int(ysize / n_window)
    margin = 100 # margin for window size
    minpx = 50 # minimal ammount of changed pixels

    # Nonzero pixels
    nonzero = trans_img.nonzero()
    nonzeroY = np.array(nonzero[0])
    nonzeroX = np.array(nonzero[1])

    # Current position
    current_leftx = init_leftx
    current_rightx = init_rightx

    left_lane_ind = []
    right_lane_ind = []

    # Loop through windows
    for window in range(n_window):
        # Calculate window boundaries
        win_y_low = ysize - ((window + 1) * window_height) # Start at bottom of image and move upwards
        win_y_high = ysize - (window * window_height)      # Start at bottom of image and move upwards
        win_xleft_low = current_leftx - margin
        win_xleft_high = current_leftx + margin
        win_xright_low = current_rightx - margin
        win_xright_high = current_rightx + margin

        # Identify nonzeros in windows
        good_left_inds = ((nonzeroY >= win_y_low) & (nonzeroY <= win_y_high) & (nonzeroX >= win_xleft_low) & (nonzeroX <= win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroY >= win_y_low) & (nonzeroY <= win_y_high) & (nonzeroX >= win_xright_low) & (nonzeroX <= win_xright_high)).nonzero()[0]
        left_lane_ind.append(good_left_inds)
        right_lane_ind.append(good_right_inds)

        # If enough pixels identified, change current position
        if (len(good_left_inds) >= minpx):
            current_leftx = np.int(sum(nonzeroX[good_left_inds])/len(nonzeroX[good_left_inds]))
        if (len(good_right_inds) >= minpx):
            current_rightx = np.int(sum(nonzeroX[good_right_inds])/len(nonzeroX[good_right_inds]))

    # Concatente arrays of indices
    left_lane_ind = np.concatenate(left_lane_ind)
    right_lane_ind = np.concatenate(right_lane_ind)

    # Extract left and right line pixel positions
    leftx = nonzeroX[left_lane_ind]
    lefty = nonzeroY[left_lane_ind] 
    rightx = nonzeroX[right_lane_ind]
    righty = nonzeroY[right_lane_ind]

    # Add pixels to line
    # left_line.allx = leftx
    # left_line.ally = lefty
    # right_line.allx = rightx
    # right_line.ally = righty

    return leftx, lefty, rightx, righty

    #return out_img, left_line, right_line

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    return left_fit, right_fit, ploty

def fit_poly_pixel(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    return left_fit, right_fit, ploty

def find_pixels_by_prior(trans_img, old_leftx, old_lefty, old_rightx, old_righty):
    # Parameters
    margin = 100 # margin for window size

    recent_left_fit, recent_right_fit, ploty = fit_poly_pixel(trans_img.shape, old_leftx, old_lefty, old_rightx, old_righty)

    # Nonzero pixels
    nonzero = trans_img.nonzero()
    nonzeroY = np.array(nonzero[0])
    nonzeroX = np.array(nonzero[1])

    # Gather pixels from prior
    left_lane_ind = ((nonzeroX > (recent_left_fit[0]*(nonzeroY**2) + recent_left_fit[1]*nonzeroY + 
                    recent_left_fit[2] - margin)) & (nonzeroX < (recent_left_fit[0]*(nonzeroY**2) + 
                    recent_left_fit[1]*nonzeroY + recent_left_fit[2] + margin)))
    right_lane_ind = ((nonzeroX > (recent_right_fit[0]*(nonzeroY**2) + recent_right_fit[1]*nonzeroY + 
                    recent_right_fit[2] - margin)) & (nonzeroX < (recent_right_fit[0]*(nonzeroY**2) + 
                    recent_right_fit[1] * nonzeroY + recent_right_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzeroX[left_lane_ind]
    lefty = nonzeroY[left_lane_ind] 
    rightx = nonzeroX[right_lane_ind]
    righty = nonzeroY[right_lane_ind]

    return leftx, lefty, rightx, righty

def calc_curvature(trans_img, leftx, lefty, rightx, righty):

    # Generate x and y values for plotting
    left_fit, right_fit, ploty = fit_poly(trans_img.shape, leftx, lefty, rightx, righty)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## Curvature calculation
    y_eval = np.max(ploty) * ym_per_pix # y at which radius is calculated

    # Left and right lane curvature calculation
    left_curvature = np.sqrt((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**3)/np.abs(2*left_fit[0])
    right_curvature = np.sqrt((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**3)/np.abs(2*right_fit[0])

    ## Position calculation
    left_current_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_current_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

    ## Visualization ##
    # Create output image
    out_img = np.dstack((trans_img, trans_img, trans_img))
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, [left_fit, left_fitx, left_curvature, left_current_x], [right_fit, right_fitx, right_curvature, right_current_x]

# def calc_curvature(trans_img, left_line, right_line):
#     """
#     Function calculates the curvature and plots it on the binary image
#     """
#     #out_img, left_line_pixels, right_line_pixels = find_lane_line_pixels(trans_img, left_line, right_line)

#     # Save recent poly fit
#     if not (len(left_line_pixels.current_fit) == 0):#[np.array([False])]):
#         left_line_pixels.recent_coeff = left_line_pixels.current_fit
#     if not (len(right_line_pixels.current_fit) == 0):# == [np.array([False])]):
#         right_line_pixels.recent_coeff = right_line_pixels.current_fit

#     # Generate x and y values for plotting
#     left_fit, right_fit, ploty = fit_poly(out_img.shape, left_line_pixels.allx, left_line_pixels.ally, right_line_pixels.allx, right_line_pixels.ally)
#     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

#     # Image dimensions
#     ysize = trans_img.shape[0]
#     xsize = trans_img.shape[1]

#     # Save new poly fit
#     left_line_pixels.current_fit = left_fit
#     right_line_pixels.current_fit = right_fit

#     # Save current x values for plotting
#     left_line_pixels.fitX = left_fitx
#     right_line_pixels.fitX = right_fitx

#     ## Visualization ##
#     # Colors in the left and right lane regions
#     out_img[lefty, leftx] = [255, 0, 0]
#     out_img[righty, rightx] = [0, 0, 255]

#     ## Curvature calculation
#     y_eval = np.max(ploty) * ym_per_pix # y at which radius is calculated

#     # Left and right lane curvature calculation
#     left_curvature = np.sqrt((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**3)/np.abs(2*left_fit[0])
#     right_curvature = np.sqrt((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**3)/np.abs(2*right_fit[0])

#     left_line_pixels.radius_of_curvature = left_curvature
#     right_line_pixels.radius_of_curvature = right_curvature

#     ## Position calculation
#     left_line_pixels.current_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
#     right_line_pixels.current_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

#     return out_img, left_line_pixels, right_line_pixels

def get_lane_curvature(left_line, right_line):
    """
    Function returns the lane curvature as mean of the left and right lane curvature
    """
    return np.mean((left_line.radius_of_curvature, right_line.radius_of_curvature))

def get_car_offset(width, left_line, right_line):
    """
    Function calculates the car offset from the lane center
    """
    # x position of lines at bottom of image (at cars position)
    right_line_x = right_line.current_x
    left_line_x = left_line.current_x

    # calculate x-value of center of lane at bottom of image (at cars position)
    lane_center_x = (right_line_x+left_line_x)/2
    
    # Offset calculation
    car_center = width * xm_per_pix / 2
    car_offset = car_center - (lane_center_x)

    return car_offset

def warp_back_results(lane_img, undistorted_img, Minv, left_line, right_line):
    """
    Function plots the results back on the original image
    INPUT:  lane_image (from calc_curvature()), 
            undistort_img (from undistortImage()), 
            Minv (inversed undistortion matrix),
            left_line (Line() instance for left line),
            right_line (Line() instance for right line)
    """
    # get radius of lane and offset of car
    radius = get_lane_curvature(left_line, right_line)
    offset = get_car_offset(undistorted_img.shape[1], left_line, right_line)

    # x and y values for plotting
    ploty = np.linspace(0, (undistorted_img.shape[0]-1), undistorted_img.shape[0]) / ym_per_pix
    left_fitx = left_line.fitX / xm_per_pix
    right_fitx = right_line.fitX / xm_per_pix

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    lane_draw_image = np.zeros_like(lane_img)
    cv2.fillPoly(lane_draw_image, np.int_([pts]), (0,255, 0))
    lane_img = cv2.addWeighted(lane_img, 1, lane_draw_image, 0.3, 0)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(lane_img, Minv, (undistorted_img.shape[1], undistorted_img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 1, 0)


    ## Write text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    line1 = "Radius of curvature: " + str(np.around(radius, decimals=2)) + "m"
    if offset < 0:
        line2 = "Vehicle is " + str(np.abs(np.around(offset, decimals=2))) + "m left of center"
    else:
        line2 = "Vehicle is " + str(np.abs(np.around(offset, decimals=2))) + "m right of center"
    cv2.putText(result, line1, (50,80), font, 2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, line2, (50,150), font, 2, (255,255,255), 2, cv2.LINE_AA)

    return result