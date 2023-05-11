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

def divide_image(image, num_divisions):
    height, width, _ = image.shape

    # Calculate the dimensions of each piece
    piece_height = height // num_divisions
    piece_width = width // num_divisions

    pieces = []
    for i in range(num_divisions):
        for j in range(num_divisions):
            # Calculate the starting and ending coordinates of the piece
            start_y = i * piece_height
            end_y = start_y + piece_height
            start_x = j * piece_width
            end_x = start_x + piece_width

            # Extract the piece from the image
            piece = image[start_y:end_y, start_x:end_x]
            pieces.append(piece)

            # Process or save each piece as needed
            # For example, you can display the piece
            cv2.imshow(f"Piece {i*num_divisions+j+1}", piece)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return pieces

# Load the image
image = cv2.imread("your_image.jpg")

# Divide the image into 4 equal-sized pieces
num_divisions = 2
divided_pieces = divide_image(image, num_divisions)
In this example, the image is divided into num_divisions x num_divisions equal-sized pieces. You can modify the code according to your specific requirements, such as the number of divisions or the file path of the image you want to process.








def check_mark(cnt, gray_square):

    zero_like = np.zeros_like(gray_square)
    # Apply the mask to the input image
    cv2.drawContours(zero_like, [cnt], 0, (255, 255, 255), -1)
    canvas = cv2.bitwise_and(gray_square, zero_like)

    #mask2 = cv2.inRange(img, 0, 0)
    #masked = cv2.bitwise_and(canvas, canvas, mask=mask2)
    #cv2.imshow(f'Contour a', canvas)
    #cv2.waitKey(0)

    return np.count_nonzero(canvas) == 0

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

    mask = np.zeros_like(img)

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(img, mask)

    contour_images.append(result)

print(len(contour_images))
for i, cimage in enumerate(contour_images[::-1]):
    playing_fileds = find_playing_fields(cv2.cvtColor(cimage, cv2.COLOR_BGR2GRAY))
    print(len(playing_fileds))
    cv2.drawContours(cimage,
                     playing_fileds,
                     -1, (0, 255, 0), 3)
    cv2.imshow(f'Contour {i}', cimage)

cv2.waitKey(0)
cv2.destroyAllWindows()
