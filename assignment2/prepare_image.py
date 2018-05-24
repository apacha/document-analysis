from PIL import Image
import cv2
import numpy as np


def binarize_image(img_path):
    #read image
    img = cv2.imread(img_path)

    #convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #binarize image and flip the foreground and background (= white text and black background) by thresholding image
    retval, thresImg = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    return thresImg

#TODO: proof if this function is needed!
def rotate_image(img):
    # detect all white pixels to find rectangle with min. area -> get rectangle with text
    area = cv2.findNonZero(img)
    rect = cv2.minAreaRect(area) #returns angle in range [-90,0)
    (x, y), (w, h), angle = rect

    # rotate img
    centerRect = (img.shape[1]//2,img.shape[0]//2)
    rotation_matrix = cv2.getRotationMatrix2D(centerRect, angle, 1);
    return cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

# Projection Profiles method - histogram profile
# mark the start and end line of each text line and return the image and line indeces
def find_text_lines(img):
    # reduce image to histogram profile
    #horizontalHist = cv2.reduce(img,0,cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    verticalHist = cv2.reduce(img, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # find all lines
    line = False
    lineHist = np.zeros((verticalHist.size,1))

    for x in range(0,verticalHist.size-1):
        if(verticalHist[x] > 0 and not line):
            lineHist[x] = 1;
            line = True

        if(verticalHist[x] == 0 and line):
            lineHist[x] = -1;
            line = False


    if(line):
        lineHist[verticalHist.size-1] = -1;

    # draw lines to mark text lines
    startLine = np.argwhere(lineHist==1)
    endLine = np.argwhere(lineHist==-1)
    lineWidth = endLine[:,0]-startLine[:,0]
    lines = []

    for x in range(0,lineWidth.size-1):
        if(np.amax(lineWidth)<=lineWidth[x]*1.25):
            img = cv2.line(img, (0, startLine[x,0]), (img.shape[1], startLine[x,0]), (255, 0, 0), 1)
            img = cv2.line(img, (0, endLine[x,0]), (img.shape[1], endLine[x,0]), (255, 0, 0), 1)
            lines = np.append(lines,[startLine[x,0],endLine[x,0]])

    return [img, lines]


if __name__ == "__main__":
    binarized_image = binarize_image("testimage.png")
    rotated_image = rotate_image(binarized_image)
    #cv2.imwrite('gray_img_transform.png', rotated_image) TODO: delete row

    [img_with_lines, lineIndex] = find_text_lines(rotated_image)

    cv2.imshow("bin", binarized_image)
    cv2.imshow("lined", img_with_lines)
    cv2.waitKey(0)