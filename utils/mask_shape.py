import cv2
import numpy as np


class Mask_Shape:
    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.contours, _ = cv2.findContours(self.gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contour = self.contours[0]
        self.peri = cv2.arcLength(self.contour, True)

    def is_approx_rectangular(self):
        approx = cv2.approxPolyDP(self.contour, 0.02 * self.peri, True)
        return len(approx) == 4

    def is_approx_triangular(self):
        approx = cv2.approxPolyDP(self.contour, 0.02 * self.peri, True)
        return len(approx) == 3

    def is_approx_circle(self):
        approx = cv2.approxPolyDP(self.contour, 0.02 * self.peri, True)
        # print('approx for circle', len(approx))
        if len(approx) > 8:
            area = cv2.contourArea(approx)
            perimeter = cv2.arcLength(approx, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            print('approx', len(approx), 'circularity', circularity)
            if circularity > 0.8:
                return True
        return False

    # def get_circle_area(self):


if __name__ == '__main__':
    # 读取二值图像
    image = cv2.imread('173.png')
    print(image.shape)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_shape = Mask_Shape(image)

    if mask_shape.is_approx_rectangular():
        print("Approximately rectangular shape found!")
    elif mask_shape.is_approx_triangular():
        print("Approximately triangular shape found!")