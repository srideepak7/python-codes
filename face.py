import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Open a video file or capture device.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find faces
    results = face_detection.process(image)

    # Draw face detections
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Display the output
    cv2.imshow('MediaPipe Face Detection', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
