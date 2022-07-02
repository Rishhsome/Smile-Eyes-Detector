import cv2

smile_tracker = cv2.CascadeClassifier('haarcascade_smile.xml')

face_tracker = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_tracker = cv2.CascadeClassifier('haarcascade_eye.xml')

video = cv2.VideoCapture(0)

while True:
    successful_frame_capture, frame = video.read()

    if successful_frame_capture:
        img_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    face_coordinates = face_tracker.detectMultiScale(img_bw)

   # smile_coordinates = smile_tracker.detectMultiScale(img_bw, 1.7, 20)  # 1.7 is the scale factor (how much blurr we want in the image) and 20 is min-neighbours which tells it to leave all the spaces where rectangle count is less than 20

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        #IT COVERS THE WHOLE FACE RECTANGLE
        the_face = frame[y : y + h, x : x + w]

        #grayscaling just the face
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #finding the simle coordinates inside the face
        smile_coordinates = smile_tracker.detectMultiScale(face_grayscale, 1.7, 20)

        eye_coordinates = eye_tracker.detectMultiScale(face_grayscale, 1.3, 5)

        for(xe, ye, we, he) in eye_coordinates:
            cv2.circle(the_face, ((xe + int(we/2)), (ye + int(he/2))), int(he/2), (255, 255, 255), 2)

        for (a, b, c, d) in smile_coordinates:
            cv2.ellipse(the_face, ((a + int(c/2)), (b + int(d/2))), (int(c/2), int(d/2)), 0, 0, 360, (0, 255, 0), 1)

        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y + h + 40), fontScale = 1, fontFace = cv2.FONT_HERSHEY_TRIPLEX, color = (0, 0, 255))
    

    cv2.imshow('Smile & Eye Detector App - Press E to EXIT', frame)

    key = cv2.waitKey(1)

    if key == 69 or key == 101:
        break

video.realease()
cv2.destroyAllWindows()
