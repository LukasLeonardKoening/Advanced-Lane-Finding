#Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = hsv[:,:,2]
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y')
    # 3) Take the absolute value of the derivative or gradient
    absSobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    uint8_sobel = np.uint8(255*absSobel/np.max(absSobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(uint8_sobel)
    # 6) Return this mask as your binary_output image
    grad_binary[(uint8_sobel > thresh[0]) & (uint8_sobel < thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
     # 1) Convert to grayscale
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = hsv[:,:,2]
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255*magnitude/np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # 1) Convert to grayscale
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = hsv[:,:,2]
    # 2) Take the gradient in x and y separately
    xSobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    ySobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    x_abs = np.absolute(xSobel)
    y_abs = np.absolute(ySobel)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(y_abs, x_abs)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return dir_binary

image = mpimg.imread("test_images/test10.jpg")

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 100)) #20, 50
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(40, 100)) #20, 50
mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(40, 100))
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.1)) # np.pi/4, 4*np.pi/10

# Image size
xsize = image.shape[1]
ysize = image.shape[0]

# Thresholds
lower_s_threshold = 100
upper_s_threshold = 255
lower_v_threshold = 180
upper_v_threshold = 255

# Color conversions
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
s_hls = hls[:,:,2]
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
v = hsv[:,:,2]

# white mask
h = hls[:,:,0]
l = hls[:,:,1]
white_mask = (l > 200) & (l <= 255)
yellow_mask = (h > 14) & (h <= 25)

# Color thresholds
s_threshold = (s_hls > lower_s_threshold) & (s_hls <= upper_s_threshold)
v_threshold = (v > lower_v_threshold) & (v <= upper_v_threshold)
#color_based_threshold = (white_mask | yellow_mask)
color_based_threshold = s_threshold & (white_mask | yellow_mask)

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | color_based_threshold] = 1
#combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#combined[((grady == 1) & (gradx == 1))] = 1
#combined[((mag_binary == 1))] = 1
#combined[((mag_binary == 1) & (dir_binary == 1))] = 1
#combined[color_based_threshold] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()