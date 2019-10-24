# IMPORTS
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import glob
import helpers

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
    lower_s_threshold = 180
    upper_s_threshold = 255

    # Color conversions
    hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    s_hls = hls[:,:,2]

    # Color thresholds
    color_based_threshold = (s_hls > lower_s_threshold) & (s_hls <= upper_s_threshold)

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