import cv2
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_label_info(label_file):
    sd = re.compile("([\w\d.]+)")
    label_format =["obj_type", "truncated", "occluded", "angle", "left", "top", "right", "bottom", "height", "width", "length", "x", "y", "z", "rotation"]
    label_list = list()
    while 1:
        d = label_file.readline()
        if d == "":
            break
        result = sd.findall(d)
        object_type = result[0]
        label_info = {label: value for label, value in zip(label_format, result)}
        for label in label_info:
            if label != "obj_type":
                label_info[label] = float(label_info[label])

        label_list.append(label_info)

    return label_list


def draw_sample(img_path, label_root_path):
    img_list = glob.glob(f"{img_path}\*.png")
    for img_path in img_list:
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        img_name = img_name[:-4]
        label_path = f"{label_root_path}\{img_name}.txt"
        label_file = open(label_path, "r", newline="")
        label_list = parse_label_info(label_file)

        for label_info in label_list:
            if label_info["obj_type"] =="DontCare":
                obj_color = (255,0,0)
            else:
                obj_color = (0,255,0)

            cv2.rectangle(img, (int(label_info["left"]), int(label_info["top"])), (int(label_info["right"]), int(label_info["bottom"])), obj_color)
        cv2.imshow("img",img)
        cv2.waitKey(45)


def visualize_detections(image, boxes, classes, scores, color=(255, 0, 0)):

    """Visualize Detections"""
    for box, _cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0))
        cv2.putText(image, _cls, (x1, y1), 2, 0.5, color, 1,)
    cv2.imshow("img", image)
    cv2.waitKey(45)
    return image



