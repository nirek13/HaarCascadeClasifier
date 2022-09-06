# Importing Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Importing Cascades
face_cascade = cv2.CascadeClassifier('Cascade_Files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascade_Files/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Cascade_Files/haarcascade_smile.xml')
sign_cascade = cv2.CascadeClassifier('Cascade_Files/sign_cascade.xml')
traffic_light_cascade = cv2.CascadeClassifier('Cascade_Files/traffic_light.xml')
hand_cascade = cv2.CascadeClassifier('Cascade_Files/hand_cascde.xml')
palm_cascade = cv2.CascadeClassifier('Cascade_Files/rpalm_cascade.xml')
yield_sign_cascade = cv2.CascadeClassifier('Cascade_Files/yield_sign_cascade.xml')
traffic_light_cascade = cv2.CascadeClassifier('Cascade_Files/traffic_light_cascade.xml')

# Making function to detect 
def detect(gray, frame):
    signs = sign_cascade.detectMultiScale(gray, 1.3, 5)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    traffic_lights = traffic_light_cascade.detectMultiScale(gray, 1.3, 5)
    palm = palm_cascade.detectMultiScale(gray, 1.3, 5)
    yield_sign = yield_sign_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)
    traffic_light = traffic_light_cascade.detectMultiScale(gray, 1.3, 7)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.2, 12)
        for (ex, ey, ew, eh) in eyes:
            cv2.putText(roi_color, 'Eyes', (ex, ey), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.putText(roi_color, 'Smile', (sx, sy), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    for (x, y, w, h) in signs:
        cv2.putText(frame, 'Signs', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
  
    for (x, y, w, h) in traffic_lights:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
        cv2.putText(frame, 'Traffic_Light', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
  
    for (x, y, w, h) in hands:
        cv2.putText(frame, 'Hand', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
  
    for (x, y, w, h) in palm:
        cv2.putText(frame, 'Palm', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
  
    for (x, y, w, h) in yield_sign:
        cv2.putText(frame, 'Yield_Sign', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)

    for (x, y, w, h) in traffic_light:
        cv2.putText(frame, 'Traffic_Light', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)

    return frame

# Stating Camera
video_capture = cv2.VideoCapture(0) # 0, if you have internal camera, 1 if you have externel camera

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Also An Edge Detector in another  frame
    lower = 115
    upper = 235
    canvas = cv2.Canny(gray, lower, upper)

    cv2.imshow('Canny Edge Detector', canvas)
    detected = detect(gray, frame)
    cv2.imshow('Video', detected)

    # Press q to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()




