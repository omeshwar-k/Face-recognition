import cv2
import numpy as np
import face_recognition
import os

# Path to folder with training images
path = 'faces'
images = []
classNames = []

# Load images and their names
for img_name in os.listdir(path):
    image = cv2.imread(f'{path}/{img_name}')
    if image is None:
        continue
    images.append(image)
    classNames.append(os.path.splitext(img_name)[0])

print("Loaded classes:", classNames)

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

# Encode known faces
knownEncodes = findEncodings(images)
print("Encoding Complete")

# Scale factor for faster processing
scale = 0.25
box_multiplier = 1 / scale

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    if not ret:
        print("Failed to grab frame")
        break
    

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(knownEncodes, face_encoding)
        face_distances = face_recognition.face_distance(knownEncodes, face_encoding)
        name = "Unknown"

        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = classNames[best_match_index].upper()

        # Scale face locations back to original frame size
        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = (int(y1*box_multiplier), int(x2*box_multiplier),
                          int(y2*box_multiplier), int(x1*box_multiplier))

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.rectangle(frame, (x1, y2-20), (x2, y2), (0,255,0), cv2.FILLED)
        cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)

    # Show the webcam feed
    cv2.imshow("Webcam Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
