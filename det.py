import cv2
import os

# Ensure folder exists
os.makedirs('faces', exist_ok=True)

# Load Haar cascade
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Open webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

name = input("Enter name to save image as: ")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # press 'c' to capture
        filename = f'faces/{name}.jpg'
        cv2.imwrite(filename, frame)
        print("✅ Image Saved:", filename)

    elif key == ord('q'):  # press 'q' to quit
        print("Exiting...")
        break

video_capture.release()
cv2.destroyAllWindows()
    