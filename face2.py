import cv2
import os
import numpy as np
import face_recognition

# Directory where the captured images are stored
captured_images_folder = 'captured_images'

# Load the face embeddings and associated person names from the stored images
known_faces = {}

def load_known_faces():
    for person_folder in os.listdir(captured_images_folder):
        person_name = person_folder
        person_folder_path = os.path.join(captured_images_folder, person_folder)

        embeddings = []
        for image_file in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_file)
            image = face_recognition.load_image_file(image_path)

            # Get the face encoding using the face_recognition library
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:
                embeddings.append(face_encoding[0])

        if embeddings:
            known_faces[person_name] = embeddings
        else:
            print(f"Warning: No faces found for {person_name}. Skipping.")

def recognize_faces(image):
    # Find face locations in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    recognized_names = []

    for face_encoding in face_encodings:
        # Compare the captured encoding with stored encodings
        min_distance = float('inf')
        recognized_name = "Unknown"

        for person_name, embeddings in known_faces.items():
            for stored_encoding in embeddings:
                dist = face_recognition.face_distance([stored_encoding], face_encoding)[0]
                if dist < min_distance:
                    min_distance = dist
                    recognized_name = person_name

        recognized_names.append(recognized_name)

    return recognized_names

if __name__ == '__main__':
    # Load the stored face embeddings and associated person names
    load_known_faces()

    # Start capturing from the device's camera
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        # Find and recognize faces in the frame
        recognized_names = recognize_faces(frame)

        for (top, right, bottom, left), recognized_name in zip(face_recognition.face_locations(frame), recognized_names):
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display the recognized person's name
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, recognized_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Stop face recognition when 'q' key is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    capture.release()
    cv2.destroyAllWindows()
