
#  cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Compute the cumulative distribution function
cdf = hist.cumsum()

# Normalize the CDF
cdf_normalized = cdf / cdf.max()

# Compute the new intensity mapping function
img_equalized = np.interp(sharpened.flatten(), bins[:-1], cdf_normalized)

# Reshape the equalized image
img_equalized = img_equalized.reshape(sharpened.shape)

# Convert back to uint8
img_equalized = (img_equalized * 255).astype(np.uint8)

blur = cv2.GaussianBlur(img_equalized, (5, 5), 0)
cv2.imshow('Contours', blur)
cv2.waitKey(0)

# Apply a binary threshold to the grayscale image
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

non_filled_rectangles = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200000000:
        x, y, w, h = cv2.boundingRect(cnt)
        non_filled_rectangles.append((x, y, w, h))

for rect in non_filled_rectangles:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Non-filled Rectangles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw the contours on the original image
#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Display the result
#cv2.imshow('Contours', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological transformations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours in the image
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
