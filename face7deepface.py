import cv2
import os
import face_recognition
import numpy as np
from deepface import DeepFace

# Directory where the captured images are stored
captured_images_folder = 'captured_images'

# Load the face embeddings and associated person names from the stored images
known_faces = {}
face_names = []

def load_known_faces():
    for person_folder in os.listdir(captured_images_folder):
        person_name = person_folder
        person_folder_path = os.path.join(captured_images_folder, person_folder)

        for image_file in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_file)
            image = face_recognition.load_image_file(image_path)

            # Get the face encoding using the face_recognition library
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:
                # Convert the numpy array to a tuple (to make it hashable)
                face_encoding_tuple = tuple(face_encoding[0])
                known_faces[face_encoding_tuple] = person_name
                face_names.append(person_name)

def recognize_faces(image):
    # Find face locations and face encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    recognized_names = []

    for face_encoding in face_encodings:
        # Convert the numpy array to a tuple (to make it hashable)
        face_encoding_tuple = tuple(face_encoding)

        # Compare the face encoding with known face encodings
        matches = [np.array_equal(face_encoding_tuple, known_face) for known_face in known_faces.keys()]
        name = "Unknown"

        if True in matches:
            # Find the best match and get the name
            match_index = matches.index(True)
            name = known_faces[list(known_faces.keys())[match_index]]

        recognized_names.append(name)

    return recognized_names, face_locations

if __name__ == '__main__':
    # Load the stored face embeddings and associated person names
    load_known_faces()

    # Start capturing from the device's camera
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        # Find and recognize faces in the frame
        recognized_names, face_locations = recognize_faces(frame)

        for (top, right, bottom, left), recognized_name in zip(face_locations, recognized_names):
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display the recognized person's name
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, recognized_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Use DeepFace for face analysis (verification)
        deepface_results = DeepFace.analyze(img_path=frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)

        # Display the analysis results for the first face (assuming only one face is in the frame)
        emotion = deepface_results[0]['emotion']
        age = deepface_results[0]['age']
        gender = deepface_results[0]['gender']
        cv2.putText(frame, f"Emotion: {emotion}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Age: {age}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Stop face recognition when 'q' key is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release
