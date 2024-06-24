import cv2
import numpy as np
import helpers

webcam = False
path = 'D:\\PROJECTS\\OpenCV-Python-Object-Measurement\\Using A4 Sheet\\1.jpg'

# cap = cv2.VideoCapture('http://192.168.90.106:8080/video')
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
width_a4_paper = 210
height_a4_paper = 297
scale = 3


while True:
    # reading image frame from live feed
    if webcam:
        success, img = cap.read()
    # reading img from static file
    else:
        img = cv2.imread(path)

    img, contours = helpers.getObjectContours(
        img, minimum_area=50000, filter=4)

    if len(contours) != 0:
        biggest_contour = contours[0][2]
        img_warp = helpers.warp_image(
            img, biggest_contour, width_a4_paper, height_a4_paper)
        cv2.imshow('Warp Image', img_warp)

    # resizing the img
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
