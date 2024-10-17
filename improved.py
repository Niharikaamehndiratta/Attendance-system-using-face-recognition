import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import os

# Create an attendance CSV file for the day
def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f'attendance_{date_str}.csv'

    # Check if file exists
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Check if the name is already recorded
    if name not in df['Name'].values:
        now = datetime.now()
        new_entry = pd.DataFrame({
            "Name": [name],
            "Date": [now.strftime("%Y-%m-%d")],
            "Time": [now.strftime("%H:%M:%S")]
        })
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(filename, index=False)

# Load known faces
known_face_encodings = []
known_face_names = []

# Load images and encode faces
def load_known_faces():
    known_faces = {
        "Ronaldo": "ronaldo.jpg",
        "Messi": "messi.jpg",
        "Benzema": "benzema.jpg",
    }

    for name, filename in known_faces.items():
        if os.path.isfile(filename):
            image = face_recognition.load_image_file(filename)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                known_face_names.append(name)
                known_face_encodings.append(face_encodings[0])
            else:
                print(f"No face found in image {filename}")
        else:
            print(f"File not found: {filename}")

# Main function
def main():
    load_known_faces()

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame")
            break

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)

        # Check if any faces were found
        if face_locations:
            # Find face encodings for the detected faces
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:  # Check if any encodings were found
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    # Mark attendance
                    mark_attendance(name)

                    # Draw a box around the face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
