import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os
import time
# import json
import requests
import uuid

# Variable to track the time of the last API call
last_api_call_times = {}
API_ENDPOINT='http://localhost:8000/api/face_data'


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

def api_call(face_name):
    device_id = uuid.uuid1()
    print("Device ID:", device_id)

    # Serialize the device ID as JSON
    # Convert UUID to string
    device_id_final = str(device_id)
    datetime_stamp = int(time.time())  # UTC/GMT timestamp

    payload = {
                "device_id": device_id_final,
                "face_id": face_name,
                "datetime_stamp": datetime_stamp,
                "face_name": face_name
            }
    print(payload)
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        print(response.text)
    except:
        print("here in exception api call")
    

# Function to recognize faces
def recognize_faces(image, knn_model, confidence_threshold=0.6):
    global last_api_call_time

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

    # Initialize the text-to-speech engine
    # engine = pyttsx3.init()


    while True:
        ret, frame = capture.read()

        # Find and recognize faces in the frame
        recognized_names = recognize_faces(frame, knn_model)
    
        door_locked = True
        recognized_name_temp = ''; 

        for recognized_name in recognized_names:
            # Display the recognized person's name
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, recognized_name, (10, 30), font, 1.0, (0, 255, 0), 2)
            print('Door Unlocked for '+recognized_name)
            recognized_name_temp = recognized_name
            door_locked = False
        
        if door_locked:
            # If no recognized names, lock the door
            print('Door Locked')
            # Speak "Door Locked"
            # engine.say("Door Locked")
            last_api_call_time = time.time()  # Update the last API call time

        else:
            if recognized_name_temp not in known_faces:
                api_call(recognized_name_temp)
                last_api_call_times[recognized_name_temp] = time.time()
                known_faces[recognized_name_temp] = True
                print('Firing API for ' + recognized_name_temp)

            else:
                current_time = time.time()
                last_api_call_time = last_api_call_times.get(recognized_name_temp, 0)

                if current_time - last_api_call_time >= 60:
                    api_call(recognized_name_temp)
                    last_api_call_times[recognized_name_temp] = current_time
                    print('Firing API for ' + recognized_name_temp)


        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Stop face recognition when 'q' key is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    capture.release()
    cv2.destroyAllWindows()
