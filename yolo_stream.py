import math

from ultralytics import YOLO
import cv2
import cvzone

# open webcam input
cap = cv2.VideoCapture("https://mediaserver.border.gov.md:50793/hls/cahul_intrare/index.m3u8")

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant","bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush"]

# create the model object with the specified weights
model = YOLO('yolo-weights/yolov8l.pt')

# define the resolution
new_width = 1280
new_height = 720

while True:
    ret, frame = cap.read()
    # check if a frame was successfully captured
    if not ret:
        break

    # resize the video
    frame = cv2.resize(frame, (new_width, new_height))

    # apply the model to the frame
    results = model(frame, stream=True)
    # draw all the boxes and labels
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1))

            # confidence and class name
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{class_names[cls]} {conf}', (max(0, x1), max(35, y1)))

    cv2.imshow("Image", frame)
    cv2.waitKey(1)
