import numpy as np
import cv2 as cv

cap = cv.VideoCapture('face.mp4')
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
mouth_mask = cv.imread('mask.png', cv.IMREAD_UNCHANGED)
while cap.isOpened():
    check, frame = cap.read()
    if check:
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        scaleFactor = 1.1
        minNeighber = 3
        face_detect = face_cascade.detectMultiScale(gray_img, scaleFactor, minNeighber)
        for (x, y, w, h) in face_detect:
            resized_mask = cv.resize(mouth_mask, (w, h), interpolation=cv.INTER_AREA)

            mask_y = y + int(0.6 * h)  
            mask_x = x
            mask_height, mask_width, _ = resized_mask.shape

            if mask_y + mask_height > frame.shape[0]:
                mask_y = frame.shape[0] - mask_height

            roi = frame[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - resized_mask[:, :, 3] / 255.0) + \
                                resized_mask[:, :, c] * (resized_mask[:, :, 3] / 255.0)

        cv.imshow("Output", frame)

        if cv.waitKey(20) & 0xff == ord("e"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()