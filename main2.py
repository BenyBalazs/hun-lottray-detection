import cv2

img = cv2.imread('5v.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological transformations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours in the image
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter out the contours based on size and aspect ratio
square_regions = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    if area > 200 and 0.9 < aspect_ratio < 1.1:
        square_regions.append((x, y, w, h))

# Draw bounding boxes around the identified square regions
for region in square_regions:
    x, y, w, h = region
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the original image with the bounding boxes
cv2.imshow('Marked squares', img)
cv2.waitKey(0)
