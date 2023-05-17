import copy

from ultralytics import YOLO

import utils
from sort import *
import random
import math
import time
import cv2
import cvzone

CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush"]

VEHICLE_NAMES = ["car", "motorbike", "bus", "truck"]

FRAMES_PER_SECOND = 30
RESOLUTION_COEFFICIENT = 0.9

# create the model object with the specified weights (will be downloaded if missing)
model = YOLO('yolo-weights/yolov8n.pt')

# create the sort object (used for tracking)
tracker = Sort(max_age=20, min_hits=5, iou_threshold=0.3)

# used for fps calculation
prev_frame_time = 0
new_frame_time = 0

# for speed calculation (lines are of format [x1, y1, x2, y2]) (starting numbers are for 1920x1080)
start_line_left_track = [int(i * RESOLUTION_COEFFICIENT) for i in [682, 545, 908, 548]]
end_line_left_track = [int(i * RESOLUTION_COEFFICIENT) for i in [441, 905, 918, 899]]
start_line_right_track = [int(i * RESOLUTION_COEFFICIENT) for i in [985, 905, 1461, 905]]
end_line_right_track = [int(i * RESOLUTION_COEFFICIENT) for i in [930, 549, 1160, 549]]

color_map = {}
tracker_lines = []
cars_crossing = {}
cars_coordinates = {}

frame_counter = 0

# cap = cv2.VideoCapture("https://func-streaming.azurewebsites.net/api/PlaylistFunction?"
#                        "code=u04eWVJBPHqaSjRUp2gGUL67JbyKqJp-HzczMR3B11HuAzFu98ooYw==&clientId=default")
cap = cv2.VideoCapture('content/ismail_fullhd.MOV')

# define the resolution
new_width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * RESOLUTION_COEFFICIENT)
new_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * RESOLUTION_COEFFICIENT)

# load the mask
mask = cv2.imread('assets/mask_ismail_prerecorded.png')
mask = cv2.resize(mask, (new_width, new_height))

SHIFT_Y = 288

while cap.isOpened():
    ret, work_frame = cap.read()
    # check if a frame was successfully captured
    if not ret:
        break

    # resize the video and apply mask
    work_frame = cv2.resize(work_frame, (new_width, new_height))
    # duplicate the frame so that the boxes do not interfere with the recognition
    show_frame = copy.deepcopy(work_frame)
    # apply the mask to the frame, so that only the road is analyzed
    work_frame = cv2.bitwise_and(work_frame, mask)

    # apply the model to the frame
    results = model(work_frame, stream=True)

    # numpy array for the detections (for tracking)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # confidence and class name
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if CLASS_NAMES[cls] == 'car' and conf > 0.3:
                # draw the bounding box and write the class_name and confidence
                # cvzone.putTextRect(frame, f'{class_names[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                # cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=20)
                # add the current detection for tracking
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    # show fps
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(show_frame, fps, (4, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

    # tracing
    results_tracker = tracker.update(detections)
    for result in results_tracker:
        x1, y1, x2, y2, vehicle_id = result
        # convert the coordinates to int
        x1, y1, x2, y2, vehicle_id = int(x1), int(y1), int(x2), int(y2), int(vehicle_id)
        # calculate the width and height of the boxes
        w, h = x2 - x1, y2 - y1
        # find the center coordinates of the boxes
        cx, cy = x1 + w // 2, y1 + h // 2

        # add the path to a list
        if vehicle_id not in cars_coordinates:
            # init the id in the dict with the format {id: (timestamp, [center1, center2...])},
            # center = tuple of center coordinates, for example (505, 101)
            cars_coordinates[vehicle_id] = (frame_counter, [(cx, cy)])
        else:
            cars_coordinates[vehicle_id][1].append((cx, cy))

        # get color from the map, or generate new one if the id is not present
        if vehicle_id not in color_map:
            color_map[vehicle_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        tracker_lines.append(((cx, cy), vehicle_id, frame_counter))
        cvzone.cornerRect(show_frame, (x1, y1, w, h), l=8, rt=1, colorR=(255, 0, 255))
        cvzone.putTextRect(show_frame, f'{vehicle_id}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=5)

        # if the car crossed the upper left line
        if start_line_left_track[0] < cx < start_line_left_track[2] and \
                start_line_left_track[1] - 10 < cy < start_line_left_track[3] + 10:
            if vehicle_id not in cars_crossing:
                cars_crossing[vehicle_id] = [('NW', frame_counter)]
            elif 'NW' not in [pair[0] for pair in cars_crossing[vehicle_id]]:
                cars_crossing[vehicle_id].append(('NW', frame_counter))

        # if the car crossed the lower left line
        if end_line_left_track[0] < cx < end_line_left_track[2] and \
                end_line_left_track[1] - 10 < cy < end_line_left_track[3] + 10:
            if vehicle_id not in cars_crossing:
                cars_crossing[vehicle_id] = [('SW', frame_counter)]
            elif 'SW' not in [pair[0] for pair in cars_crossing[vehicle_id]]:
                cars_crossing[vehicle_id].append(('SW', frame_counter))

        # if the car crossed the lower right line
        if start_line_right_track[0] < cx < start_line_right_track[2] and \
                start_line_right_track[1] - 10 < cy < start_line_right_track[3] + 10:
            if vehicle_id not in cars_crossing:
                cars_crossing[vehicle_id] = [('SE', frame_counter)]
            elif 'SE' not in [pair[0] for pair in cars_crossing[vehicle_id]]:
                cars_crossing[vehicle_id].append(('SE', frame_counter))

        # if the car crossed the upper right line
        if end_line_right_track[0] < cx < end_line_right_track[2] and \
                end_line_right_track[1] - 10 < cy < end_line_right_track[3] + 10:
            if vehicle_id not in cars_crossing:
                cars_crossing[vehicle_id] = [('NE', frame_counter)]
            elif 'NE' not in [pair[0] for pair in cars_crossing[vehicle_id]]:
                cars_crossing[vehicle_id].append(('NE', frame_counter))

    # print(cars_crossing)
    # print(frame_counter)
    # print(cars_coordinates)

    # check for speed in cars and remove cars that only got one gateway
    copy_cars_crossing = cars_crossing.copy()
    for k, v in copy_cars_crossing.items():
        if len(v) == 2:
            frames = v[1][1] - v[0][1]
            print(f'car with id: {k} crossed the lines in {frames} frames')
            if frames < 40:
                print(f'car with id: {k} is speeding')
            if v[0][0] not in ['NW', 'SE']:
                print(f'car with id: {k} is heading in the wrong direction')
            del cars_crossing[k]
        elif frame_counter > v[0][1] + 200:
            del cars_crossing[k]

    # remove the car from the dict if the timestamp is too old
    copy_cars_coordinates = cars_coordinates.copy()
    for k, v in copy_cars_coordinates.items():
        if v[0] + 150 < frame_counter:
            with open('cars_paths.txt', 'a') as f:
                f.write(f'{k}: {v}\n')
            del cars_coordinates[k]

    # tracker trails logic
    for tracker_line in tracker_lines:
        if tracker_line[2] + FRAMES_PER_SECOND * 2 < frame_counter:
            tracker_lines.remove(tracker_line)
            continue

        cv2.circle(
            show_frame,
            tracker_line[0],
            utils.map_value(frame_counter - tracker_line[2], 0, FRAMES_PER_SECOND * 2, 5, 0),
            color_map[tracker_line[1]],
            cv2.FILLED
        )
    frame_counter += 1

    # draw start and end lines for speed check
    # # upper left line
    # cv2.line(
    #     show_frame,
    #     (start_line_left_track[0], start_line_left_track[1]),
    #     (start_line_left_track[2], start_line_left_track[3]),
    #     (255, 255, 255),
    #     3
    # )
    #
    # # lower left line
    # cv2.line(
    #     show_frame,
    #     (end_line_left_track[0], end_line_left_track[1]),
    #     (end_line_left_track[2], end_line_left_track[3]),
    #     (255, 255, 255),
    #     3
    # )
    #
    # # lower right line
    # cv2.line(
    #     show_frame,
    #     (start_line_right_track[0], start_line_right_track[1]),
    #     (start_line_right_track[2], start_line_right_track[3]),
    #     (0, 255, 255),
    #     3
    # )
    #
    # # upper right line
    # cv2.line(
    #     show_frame,
    #     (end_line_right_track[0], end_line_right_track[1]),
    #     (end_line_right_track[2], end_line_right_track[3]),
    #     (0, 255, 255),
    #     3
    # )

    cv2.imshow("", show_frame)
    cv2.waitKey(1)
