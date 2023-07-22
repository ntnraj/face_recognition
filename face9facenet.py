import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os

# Directory where the captured images are stored
captured_images_folder = 'captured_images'

# Load the face embeddings and associated person names from the stored images
known_faces = {}
face_encodings = []
face_names = []

# Function to load known faces and their encodings
def load_known_faces():
    for person_folder in os.listdir(captured_images_folder):
        person_name = person_folder
        person_folder_path = os.path.join(captured_images_folder, person_folder)

        for image_file in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_file)

            # Load the image using OpenCV
            image = cv2.imread(image_path)

            # Convert the image to RGB (facenet-pytorch MTCNN requires RGB images)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Use MTCNN to detect face and get the face encoding
            face = mtcnn(image_rgb)

            if face is not None:
                # Convert the face tensor to numpy array
                face_encoding = resnet(face.unsqueeze(0)).detach().numpy()

                face_encodings.append(face_encoding[0])
                face_names.append(person_name)

# Function to recognize faces
def recognize_faces(image, knn_model, confidence_threshold=0.6):
    # Convert the image to RGB (facenet-pytorch MTCNN requires RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MTCNN to detect face and get the face encoding
    face = mtcnn(image_rgb)

    recognized_names = []

    if face is not None:
        # Convert the face tensor to numpy array
        face_encoding = resnet(face.unsqueeze(0)).detach().numpy()

        # Use the KNN model to find the closest match
        distances, indices = knn_model.kneighbors(face_encoding)
        min_distance = distances[0][0]

        if min_distance < confidence_threshold:
            recognized_name = face_names[indices[0][0]]
            recognized_names.append(recognized_name)

    return recognized_names

if __name__ == '__main__':
    # Create an instance of the MTCNN class using CPU
    mtcnn = MTCNN(device='cpu')

    # Create an instance of the InceptionResnetV1 class using CPU
    resnet = InceptionResnetV1(pretrained='vggface2', device='cpu').eval()

    # Load the stored face embeddings and associated person names
    load_known_faces()

    # Train the KNN model with the known face encodings
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(face_encodings, face_names)

    # Start capturing from the device's camera
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        # Find and recognize faces in the frame
        recognized_names = recognize_faces(frame, knn_model)

        for recognized_name in recognized_names:
            # Display the recognized person's name
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, recognized_name, (10, 30), font, 1.0, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Stop face recognition when 'q' key is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    capture.release()
    cv2.destroyAllWindows()
