import numpy as np
import cv2
from scipy import ndimage


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class Circle(object):
    def __init__(self, img):
        self.ix, self.iy = -1, -1
        self.jx, self.jy = -1, -1
        self.drawing = False
        self.img = img
        self.radius = 0

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img = self.img.copy()
                center_x, center_y = (self.ix + x) // 2, (self.iy + y) // 2
                radius = abs(self.ix - center_x)
                cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), thickness=1)
                cv2.imshow('image', img)
                self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            img = self.img.copy()
            self.jx, self.jy = x, y
            self.radius = abs(self.ix - self.jx) // 2
            phi = self.get_phi()
            cv2.imshow('image', mask_img_from_phi(img, phi))
            self.drawing = False

    def get_phi(self):
        x, y = np.meshgrid(np.arange(self.img.shape[1]), np.arange(self.img.shape[0]))
        phi = np.sqrt((x - (self.ix + self.jx) // 2) ** 2 + (y - (self.iy + self.jy) // 2) ** 2) - self.radius
        return phi


class Rectangle(object):
    def __init__(self, img):
        self.ix, self.iy = -1, -1
        self.jx, self.jy = -1, -1
        self.drawing = False
        self.img = img

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img = self.img.copy()
                cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 0, 255), thickness=1)
                cv2.imshow('image', img)
                self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            img = self.img.copy()
            self.jx, self.jy = x, y
            phi = self.get_phi()
            cv2.imshow('image', mask_img_from_phi(img, phi))
            self.drawing = False

    def get_phi(self):
        mask = np.ones(self.img.shape[0:2])
        mask[self.iy:self.jy, self.ix:self.jx] = 0
        phi = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(1 - mask) + mask - 0.5
        return phi


def mask_img_from_phi(img, phi):
    color = Colors()
    mask = np.zeros_like(img)
    mask[phi <= 0] = color(0)
    mask[phi > 0] = color(10)
    return cv2.addWeighted(img, 0.6, mask, 0.4, 0)


def select_region(img):
    cv2.namedWindow('image')
    cv2.setWindowTitle('image', 'Select the initial curve (Press q to finish)')
    cv2.imshow('image', img)
    circle = Circle(img)
    rectangle = Rectangle(img)
    mode = 0
    cv2.setMouseCallback('image', circle.draw_circle)

    # Loop until the user draw a circle
    while not(cv2.waitKey(1) & 0xff == ord('q')) or (not (circle.radius) and (rectangle.ix == -1)):
        if cv2.waitKey(1) & 0xff == ord('c'):
            mode = 0
            cv2.setMouseCallback('image', lambda *args: None)
            cv2.setMouseCallback('image', circle.draw_circle)
        elif cv2.waitKey(1) & 0xff == ord('r'):
            mode = 1
            cv2.setMouseCallback('image', lambda *args: None)
            cv2.setMouseCallback('image', rectangle.draw_rectangle)

    cv2.destroyAllWindows()
    if mode:
        return rectangle.get_phi()

    return circle.get_phi()


if __name__ == '__main__':
    img = cv2.imread('./testcase/pic-2.jpg')
    print(select_region(img))

