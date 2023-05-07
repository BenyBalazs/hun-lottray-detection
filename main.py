import cv2
import numpy as np


def find_rectangles(gray_image):
    # Apply Canny edge detection to the image
    canny = cv2.Canny(gray_image, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out the contours based on size and aspect ratio
    squares = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 100 and cv2.isContourConvex(approx):
            squares.append(approx)

    return squares

def find_playing_fields(gray_image):

    # cv2.imshow("asd", gray_image)
    # Apply Canny edge detection to the image

    # Define the Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Apply the Laplacian kernel to the image
    sharpened = cv2.filter2D(gray_image, -1, kernel)

    cv2.imshow('sharpened Image', sharpened)

    canny = cv2.Canny(sharpened, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    # Filter out the contours based on size and aspect ratio
    squares = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1 and cv2.isContourConvex(approx):
            squares.append(approx)

    return squares

# Load the image
img = cv2.imread('5v.bmp')

# Convert the image to grayscale

blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

winname = "Test"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)

gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
cv2.imshow(winname, gray)
cv2.waitKey(0)

squares = find_rectangles(gray)

contour_images = []

# Loop over the contours and extract each contour
for i, contour in enumerate(squares):
    # Create a mask of the same size as the input image
    mask = np.zeros_like(img)

    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    # Apply the mask to the input image
    result = cv2.bitwise_and(img, mask)

    # Add the extracted contour to the list
    contour_images.append(result)

cv2.drawContours(contour_images[-1],
                 find_playing_fields(cv2.cvtColor(contour_images[-1], cv2.COLOR_BGR2GRAY)),
                 -1, (0, 255, 0), 3)
cv2.imshow(f'Contour {1}', contour_images[-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
