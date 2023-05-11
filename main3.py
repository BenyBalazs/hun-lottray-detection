import cv2
import numpy as np

#Felosztjuk kézzel egyenletesen és utána eltüntetjük a porisat HSV-vel és utána a felosztáson megnézzük hol nem maradt valami mert a toll fekete.
def find_rectangles(gray_image):

    canny = cv2.Canny(gray_image, 50, 150)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and 10000 < cv2.contourArea(approx) < 100000000000000 and cv2.isContourConvex(approx):
            squares.append(approx)

    return squares

def divide_image(playing_field, num_divisions):

    pixels = 3
    height, width = playing_field.shape[:2]
    cropped_image = playing_field[pixels:height - pixels, pixels:width - pixels]

    cv2.imshow("field", cropped_image)
    cv2.waitKey(0)

    height, width = cropped_image.shape[:2]
    # Calculate the dimensions of each piece
    piece_height = height // num_divisions
    piece_width = width // num_divisions

    pieces = []
    for i in range(num_divisions - 1):
        for j in range(num_divisions):
            # Calculate the starting and ending coordinates of the piece
            start_y = i * piece_height
            end_y = start_y + piece_height
            start_x = j * piece_width
            end_x = start_x + piece_width

            # Extract the piece from the image
            piece = cropped_image[start_y:end_y, start_x:end_x]
            pieces.append(piece)

            # Process or save each piece as needed
            # For example, you can display the piece
            cv2.imshow(f"Piece {i*num_divisions+j+1}", piece)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    print("pices", len(pieces))
    return pieces

img = cv2.imread('5v.bmp')


blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

winname = "Test"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)

gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
cv2.imshow(winname, gray)
cv2.waitKey(0)

squares = find_rectangles(gray)

contour_images = []

for i, contour in enumerate(squares):

    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = img[y:y + h, x:x + w]
    contour_images.append(cropped_image)

divide_image(contour_images[-1], 10)

cv2.waitKey(0)
cv2.destroyAllWindows()
