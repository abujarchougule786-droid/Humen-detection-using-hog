import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.03
    )

    for i, (x, y, w, h) in enumerate(boxes):

        # Ignore weak detections
        if weights[i] < 0.7:
            continue

        cv2.rectangle(frame,
                      (x, y),
                      (x+w, y+h),
                      (0, 255, 0), 3)

        cv2.putText(frame,
                    "Human",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()