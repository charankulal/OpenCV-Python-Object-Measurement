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

scale = 3
width_a4_paper = 210*scale
height_a4_paper = 297*scale


while True:
    # reading image frame from live feed
    if webcam:
        success, img = cap.read()
    # reading img from static file
    else:
        img = cv2.imread(path)

    img_contours, contours = helpers.getObjectContours(
        img, minimum_area=50000, filter=4)

    if len(contours) != 0:
        biggest_contour = contours[0][2]
        img_warp = helpers.warp_image(
            img, biggest_contour, width_a4_paper, height_a4_paper)
        img_contours_two, contours_two = helpers.getObjectContours(
            img_warp, minimum_area=2000, filter=4, canny_threshold=[50, 50], draw=False)
        if len(contours) != 0:
            for obj in contours_two:
                cv2.polylines(img_contours_two, [obj[2]], True, (0, 255, 0), 2)
                n_points = helpers.reorder_points(obj[2])
                nW = round((helpers.find_distance(
                    n_points[0][0]//scale, n_points[1][0]//scale)/10), 1)
                nH = round((helpers.find_distance(
                    n_points[0][0]//scale, n_points[2][0]//scale)/10), 1)
                cv2.arrowedLine(img_contours_two, (n_points[0][0][0], n_points[0][0][1]), (n_points[1][0][0], n_points[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img_contours_two, (n_points[0][0][0], n_points[0][0][1]), (n_points[2][0][0], n_points[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(img_contours_two, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(img_contours_two, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv2.imshow('Warp Image', img_contours_two)

    # resizing the img
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
