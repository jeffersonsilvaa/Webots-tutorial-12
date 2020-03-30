# **************************************************************
# Project Extra - Disciplina de robótica Móvel UFC / IFCE / LAPISCO
# Simulação com Veículo e detecção de objetos com Yolo v3 - Webots R2020a
#       Veículo BMW X5 - controles básicos, Câmera
#        Python 3.5 na IDE Pycharm - controller <extern>
#                By: Jefferson Silva Almeida
#                       Data: 24/03/2020
# **************************************************************

import cv2
import numpy as np
import argparse
import time

from vehicle import Driver
from controller import Camera
import math

TIME_STEP = 32  # ms
MAX_SPEED = 100  # km/h

driver = Driver()

speedFoward = 0.1 * MAX_SPEED  # km/h
speedBrake = 0  # km/h
cont = 0
plot = 10

cameraRGB = driver.getCamera('camera')
Camera.enable(cameraRGB, TIME_STEP)


# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    return img, height, width, channels


def display_blob(blob):
    '''
		Three images each for RED, GREEN, BLUE channel
	'''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    cv2.waitKey(1)

cont = 16
while driver.step() != -1:
    driver.setCruisingSpeed(10)
    driver.setSteeringAngle(0)

    if cont > 15:
        Camera.getImage(cameraRGB)
        Camera.saveImage(cameraRGB, 'color.png', 1)
        image_path = 'color.png'
        image_detect(image_path)
        cont = 0

    cont += 1

cv2.destroyAllWindows()
