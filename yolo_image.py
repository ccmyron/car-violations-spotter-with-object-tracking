from ultralytics import YOLO
import cv2

# create the model object with the specified weights (will be downloaded if missing)
model = YOLO('yolo-weights/yolov8n.pt')
# load the image and show it at the end
results = model("content/traffic_jam_1050x700.jpg", show=True)

# wait in order to get the loaded image at the end
cv2.waitKey(0)
