import cv2
import numpy as np

# from scipy.spatial import distance as dist

# import numpy as np
# import argparse

# import cv2

img = cv2.imread("f5 .jpg")

ball_low_col = np.array([200, 200, 150])  # RGB
ball_low_col = np.flipud(ball_low_col)  # BGR

ball_top_col = np.array([250, 250, 250])
ball_top_col = np.flipud(ball_top_col)

# find balls by colot matrix of ones where color in color range
mask = cv2.inRange(img, ball_low_col, ball_top_col)
# filter immage using mask
res = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((3, 3), np.uint8)
# remove holes inside the balls
closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
# remove wrond points outside the balls and sticks
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
# find countour of the objects
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# conv gradient to monochome to use as a mask
mono_img = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
# create new mask to fully cover balls
mask = cv2.inRange(opening, ball_low_col, ball_top_col)

h, w = mask.shape[:2]
# equalize image
rect_cords = np.array([[0, 0],
                       [w - 0, 0],
                       [w - 0, h - 0],
                       [0, h - 0]], np.float32)
rect_dst = np.array([[0, 0],
                     [w + 2, 0],
                     [w + 2, h + 2],
                     [0, h + 2]], np.float32)

M = cv2.getPerspectiveTransform(rect_cords, rect_dst)
wraped = cv2.warpPerspective(mono_img, M, (w + 2, h + 2))

# print(wraped.shape, img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)


# a = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# # sort the contours from left-to-right and initialize the
# # 'pixels per metric' calibration variable
# (cnts, _) = contours.sort_contours(cnts)
# pixelsPerMetric = None


def thresh_callback(thresh_hold):
    height, width, colors = img.shape
    edges = cv2.Canny(blur, thresh_hold / 10, thresh_hold / 12)
    drawing = img_copy.copy()
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius > height / 20 and radius < height / 5:
            cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 1)
            cv2.circle(drawing, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            cv2.imshow('output', drawing)
            cv2.imshow('input', img)


# img = edged.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = mono_img.copy()

blur = cv2.GaussianBlur(gray, (5, 5), 0)
img_copy = img.copy()

# thresh_callback(0)
#
# for i, row in enumerate(img):
#     for j, color in enumerate(row):
#         # print(row, i, j, color)
#         point = j, i
# cv2.floodFill(img, mono_img, (200, 400), newVal=255)
# print(code, box.shape, morecode)

# res = cv2.bitwise_and(img, gradient)

# first try balls detected
# fin after filtration


# mono_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("res", res)
# cv2.imshow("r", a[0])
# cv2.imshow("closing", closing)
cv2.imshow("opening(erosion -> dilation)", opening)
cv2.imshow("gradient(dilation - erosion)", mono_img)
cv2.waitKey(0)