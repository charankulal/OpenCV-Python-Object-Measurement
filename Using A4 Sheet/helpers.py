import cv2
import numpy as np


def getObjectContours(img, canny_threshold=[100, 100], show_canny_img=False, minimum_area=1000, filter=0, draw=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, canny_threshold[0], canny_threshold[1])
    kernel = np.ones((5, 5))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=3)
    img_threshold = cv2.erode(img_dilate, kernel, iterations=2)
    if show_canny_img:
        cv2.imshow('Canny Output', img_threshold)

    contours, hierarchy = cv2.findContours(
        img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > minimum_area:
            perimeter = cv2.arcLength(i, True)
            approximation_var = cv2.approxPolyDP(i, 0.02*perimeter, True)
            bounding_box = cv2.boundingRect(approximation_var)
            if filter > 0:
                if len(approximation_var) == filter:
                    final_contours.append(
                        [len(approximation_var), area, approximation_var, bounding_box, i])
            else:
                final_contours.append(
                    [len(approximation_var), area, approximation_var, bounding_box, i])

    final_contours = sorted(final_contours, key=lambda x: x[1], reverse=True)
    if draw:
        for contour in final_contours:
            cv2.drawContours(img, contour[4], -1, (0, 0, 255), 3)

    return img, final_contours


def reorder_points(points):
    new_points = np.zeros_like(points)
    points = points.reshape((4, 2))
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def warp_image(img, points, w, h, padding=20):
    points = reorder_points(points)
    point_1 = np.float32(points)
    point_2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(point_1, point_2)
    img_warp = cv2.warpPerspective(img, matrix, (w, h))
    img_warp = img_warp[padding:img_warp.shape[0] -
                        padding, padding:img_warp.shape[1]-padding]
    return img_warp

def find_distance(points_1, points_2):
    return ((points_2[0]-points_1[0])**2 + (points_2[1]-points_1[1])**2)**0.5
