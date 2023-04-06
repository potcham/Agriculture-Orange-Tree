import cv2 
from ultralytics import YOLO
import math

cap = cv2.VideoCapture('demo_orange.mp4')

model = YOLO("orange-tree.pt")

classes = ['green_orange', 'orange']

BoxColor = (255, 255, 0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
output = cv2.VideoWriter('demo-orange.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)
while True:
    success, frame = cap.read()
    if success:
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                currentClass = classes[cls]

                if conf > 0.6:
                    cv2.putText(frame, text=f'{classes[cls]} {conf}',color=BoxColor,thickness=1, fontScale=0.7, org=(max(0, x1+1), max(15, y1+1)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BoxColor, 1)

        cv2.imshow("Orange Detection", frame)
        output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    

cap.release()
cv2.destroyAllWindows()