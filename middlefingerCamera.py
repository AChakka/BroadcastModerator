import cv2
import torch
from ultralytics import YOLO


model = YOLO(r"C:\Users\adith\BroadcastModerator\best.pt")
cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    results = model(frame)


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = box.conf[0].item()  
            label = model.names[int(box.cls[0])]  

            if confidence > 0.8:  
                roi = frame[y1:y2, x1:x2]  
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)  
                frame[y1:y2, x1:x2] = blurred_roi  
            else:
  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("Ruffian Detector", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
