import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from scipy.spatial import distance
import os
import dlib

# Directory where the captured images are stored
captured_images_folder = 'captured_images'

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the face detector and face recognition model from dlib
face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1("path/to/dlib_face_recognition_resnet_model_v1.dat")

def get_face_embedding(image):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the Haar Cascade classifier to detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_roi = image[y:y + h, x:x + w]

        # Convert the face_roi to dlib rectangle format
        dlib_rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)

        # Use dlib face recognition model to compute face embedding
        face_embedding = face_recognition_model.compute_face_descriptor(image, dlib_rect)
        face_embedding = np.array(face_embedding)

        return face_embedding

    return None

# Load the FaceNet model from TensorFlow Hub
face_net_model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4"
face_net_model = hub.load(face_net_model_url)

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
            face_embedding = get_face_embedding(image)  # Replace this with your function to get embeddings
            embeddings.append(face_embedding)

        known_faces[person_name] = embeddings

def recognize_face(image):
    # Compute the embedding of the captured image
    captured_embedding = get_face_embedding(image)  # Replace this with your function to get embeddings

    # Compare the captured embedding with stored embeddings
    min_distance = float('inf')
    recognized_name = "Unknown"

    for person_name, embeddings in known_faces.items():
        for stored_embedding in embeddings:
            dist = distance.euclidean(captured_embedding, stored_embedding)
            if dist < min_distance:
                min_distance = dist
                recognized_name = person_name

    return recognized_name

if __name__ == '__main__':
    # Load the stored face embeddings and associated person names
    load_known_faces()

    # Start capturing from the device's camera
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Perform face recognition when 'q' key is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            recognized_name = recognize_face(frame)
            print(f"Recognized Name: {recognized_name}")

        # Stop face recognition when 'esc' key is pressed
        elif key == 27:
            break

    # Release the capture and close the window
    capture.release()
    cv2.destroyAllWindows()
