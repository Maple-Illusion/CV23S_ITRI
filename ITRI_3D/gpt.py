import cv2
import numpy as np

# Load the image
image = cv2.imread("../../ITRI_DLC/test2/dataset/1681114219_676808715/raw_image.jpg")

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for white color in HSV
lower_white = np.array([0,0,168], dtype=np.uint8)
upper_white = np.array([172,111,255], dtype=np.uint8)

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image, image, mask=mask)

# Show the images
cv2.imshow('Original Image', image)
cv2.imshow('mask', mask)
cv2.imshow('Filtered Color Only', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
