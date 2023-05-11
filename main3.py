import cv2
import numpy as np


# Felosztjuk kézzel egyenletesen és utána eltüntetjük a porisat HSV-vel és utána a felosztáson megnézzük hol nem maradt valami mert a toll fekete.
def find_rectangles(gray_image):
    canny = cv2.Canny(gray_image, 50, 150)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and 10000 < cv2.contourArea(approx) < 100000000000000 and cv2.isContourConvex(approx):
            squares.append(approx)

    return squares


def divide_image(playing_field):
    pixels = 3
    height, width = playing_field.shape[:2]
    cropped_image = playing_field[pixels:height - pixels, pixels:width - pixels]

    height, width = cropped_image.shape[:2]
    piece_height = height // 9
    piece_width = width // 10

    pieces = []
    for i in range(9):
        for j in range(10):
            start_y = i * piece_height
            end_y = start_y + piece_height
            start_x = j * piece_width
            end_x = start_x + piece_width

            piece = cropped_image[start_y:end_y, start_x:end_x]
            pieces.append(piece)

    return pieces


def check_filled(square):
    height, width = square.shape[:2]
    white = np.ones((height, width, 3), dtype=np.uint8) * 255

    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)

    lower_range = np.array([0, 0, 0])
    upper_range = np.array([100, 200, 30])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    result = cv2.copyTo(white, mask)
    return cv2.countNonZero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)) > 30


def main():
    img = cv2.imread('5vk.bmp')

    cv2.imshow("Szelvény", img)

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    playing_field_contours = find_rectangles(gray)
    playing_fields = []

    for i, contour in enumerate(playing_field_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = img[y:y + h, x:x + w]
        playing_fields.append(cropped_image)

    for i, playing_filed in enumerate(playing_fields[::-1]):
        print(f'Játék: {i + 1}:')
        played_numbers = []
        for j, square in enumerate(divide_image(playing_filed)):
            if check_filled(square):
                played_numbers.append(j + 1)
        number_of_played_fields = len(played_numbers)
        if number_of_played_fields > 5:
            print(f'Érvénytelen túl sok mező: {number_of_played_fields}:')
        if number_of_played_fields < 5:
            print(f'Érvénytelen túl kevés mező: {number_of_played_fields}:')
        if number_of_played_fields == 5:
            print(f'Érvényes megjátszott számok: {played_numbers}:')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
