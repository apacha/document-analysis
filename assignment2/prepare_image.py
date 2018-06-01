from PIL import Image
import cv2
import numpy as np


def load_and_binarize_image(img_path):
    image = cv2.imread(img_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # TODO: Why do we binarize the image here? We don't need this and it introduced additional noise
    # binarize image and flip the foreground and background (= white text and black background) by thresholding image
    retval, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return binary_image


def find_text_lines(img):
    """
        Projection Profiles method - histogram profile
        mark the start and end line of each text line and return the image and line indeces
    :param img:
    :return:
    """
    # reduce image to histogram profile
    # horizontalHist = cv2.reduce(img,0,cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    verticalHist = cv2.reduce(img, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # find all lines
    line = False
    lineHist = np.zeros((verticalHist.size, 1))

    for x in range(0, verticalHist.size - 1):
        if (verticalHist[x] > 0 and not line):
            lineHist[x] = 1
            line = True

        if (verticalHist[x] == 0 and line):
            lineHist[x] = -1
            line = False

    if (line):
        lineHist[verticalHist.size - 1] = -1

    # draw lines to mark text lines
    startLine = np.argwhere(lineHist == 1)
    endLine = np.argwhere(lineHist == -1)
    lineWidth = endLine[:, 0] - startLine[:, 0]
    linesStart = []
    linesEnd = []
    max = 0

    for x in range(0, lineWidth.size):
        if (lineWidth[x] > 20 and lineWidth[x] < 60):
            # img = cv2.line(img, (0, startLine[x,0]), (img.shape[1], startLine[x,0]), (255, 0, 0), 1)
            # img = cv2.line(img, (0, endLine[x,0]), (img.shape[1], endLine[x,0]), (255, 0, 0), 1)
            linesStart = np.append(linesStart, [startLine[x, 0]])
            linesEnd = np.append(linesEnd, [endLine[x, 0]])
            if (lineWidth[x] > max):
                max = lineWidth[x]

    lines = np.column_stack((linesStart, linesEnd))

    return [img, lines, max]


def get_image_lines(image_path: str):
    binarized_image = load_and_binarize_image(image_path)
    [img_with_lines, lineIndex, max] = find_text_lines(binarized_image)
    # cv2.imshow("bin", binarized_image)
    # cv2.imshow("lined", img_with_lines)
    # cv2.waitKey(0)

    return binarized_image, lineIndex, max


if __name__ == "__main__":
    get_image_lines("a03-020.png")
