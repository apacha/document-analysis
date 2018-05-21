from PIL import Image
import cv2
import numpy as np


def binarize_image(img_path):
    #read image
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #binarize image = white text and black background
    retval, thresImg = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    return thresImg

def rotate_image(img):
    # detect all white pixels to find rectangle with min. area -> get rectangle with text
    area = cv2.findNonZero(img)
    rect = cv2.minAreaRect(area)
    (x, y), (w, h), angle = rect

    # ============= only for test =====================
    # box = cv2.boxPoints(rect)
    # box_d = np.int0(box)
    # cv2.drawContours(gray_img,[box_d],0,[0,255,0],3)
    #
    # cv2.imwrite('gray_testimage.png', gray_img)
    # =========== TODO: delete ===================

    # remove noise? TODO: proof if necessary

    # rotate img
    centerRect = (w//2,h//2)
    rotation_matrix = cv2.getRotationMatrix2D(centerRect, angle, 1);
    return cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

def find_text_lines(img):
    # find white pixel groups

    # draw lines to mark text lines

    binarized_image = {}
    return


if __name__ == "__main__":
    binarized_image = binarize_image("testimage.jpg")
    rotated_image = rotate_image(binarized_image)
    cv2.imwrite('gray_img_transform.png', rotated_image)
    img_with_lines = find_text_lines(rotated_image)