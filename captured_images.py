import cv2
import os

captured_images_folder = 'captured_images'

def store_faces(person_name, num_images=10):
    person_folder = os.path.join(captured_images_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    while count < num_images:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(person_folder, f'image{count}.jpg'), face_image)
            count += 1

        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    person_name = input("Enter the person's name: ")
    num_images = int(input("Enter the number of images to capture: "))
    store_faces(person_name, num_images)
