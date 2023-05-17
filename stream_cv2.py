import cv2

cap = cv2.VideoCapture("https://func-streaming.azurewebsites.net/api/PlaylistFunction?"
                       "code=u04eWVJBPHqaSjRUp2gGUL67JbyKqJp-HzczMR3B11HuAzFu98ooYw==&clientId=default")

while cap.isOpened():
    ret, work_frame = cap.read()
    # check if a frame was successfully captured
    if not ret:
        break

    delay = int(1000 / 30)

    cv2.imshow("", work_frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
