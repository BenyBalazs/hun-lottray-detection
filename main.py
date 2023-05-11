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

def find_playing_fields(gray_image):

    # cv2.imshow("asd", gray_image)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    dilate_kernel = np.ones((1, 1), np.uint8)

    sharpened = cv2.filter2D(gray_image, -1, kernel)
    img_dilation = cv2.erode(sharpened, dilate_kernel, iterations=30)

    bs = cv2.GaussianBlur(img_dilation, (5, 5), 0)

    _, thresh = cv2.threshold(bs, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and 200 < cv2.contourArea(approx) < 1000:
            squares.append(approx)

    for f, cnt in enumerate(squares[::-1]):
        check_mark(cnt, gray_image)

    return squares

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

    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = img[y:y + h, x:x + w]

    cv2.imshow("asdasd", cropped_image)
    cv2.waitKey(0)

    contour_images.append(cropped_image)

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
