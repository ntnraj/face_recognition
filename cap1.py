import cv2
import os
import numpy as np
import face_recognition

# Directory where the captured images will be stored
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
            image = cv2.imread(image_path)

            # Get the face encoding using the face_recognition library
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                embeddings.append(face_encoding)

        if embeddings:
            known_faces[person_name] = embeddings
        else:
            print(f"Warning: No faces found for {person_name}. Skipping.")

def capture_images(person_name):
    # Create a folder for the person's images
    person_folder = os.path.join(captured_images_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    print(f"Capturing images for {person_name}. Press 'q' to stop capturing.")

    capture = cv2.VideoCapture(0)
    image_count = 0

    while True:
        ret, frame = capture.read()

        # Find face locations in the frame
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Crop the face region for recognition
            face_roi = frame[top:bottom, left:right]

            # Save the captured face image
            image_path = os.path.join(person_folder, f"captured_image_{image_count}.jpg")
            cv2.imwrite(image_path, face_roi)
            image_count += 1

        # Display the frame
        cv2.imshow('Capture Images', frame)

        # Stop capturing when 'q' key is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    capture.release()
    cv2.destroyAllWindows()

def recognize_face(image):
    # Find face locations in the image
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        # No faces found, return 'Unknown'
        return "Unknown"

    # Get the face encoding of the captured image using the face_recognition library
    captured_encoding = face_recognition.face_encodings(image, face_locations)[0]

    # Compare the captured encoding with stored encodings
    min_distance = float('inf')
    recognized_name = "Unknown"

    for person_name, embeddings in known_faces.items():
        for stored_encoding in embeddings:
            dist = face_recognition.face_distance([stored_encoding], captured_encoding)[0]
            if dist < min_distance:
                min_distance = dist
                recognized_name = person_name

    return recognized_name

if __name__ == '__main__':
    # Load the stored face embeddings and associated person names
    load_known_faces()

    # Ask for the person's name and capture images
    person_name = input("Enter the person's name: ")
    capture_images(person_name)

    # Start capturing from the device's camera
    capture = cv2.VideoCapture(0)

    # Create the window and set properties for focus
    cv2.namedWindow('Face Recognition')
    cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = capture.read()

        # Find face locations in the frame
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Crop the face region for recognition
            face_roi = frame[top:bottom, left:right]

            # Perform face recognition on the face region
            recognized_name = recognize_face(face_roi)

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
