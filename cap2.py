import cv2
import os

# Directory where the captured images will be stored
captured_images_folder = 'captured_images'

# Create the directory if it doesn't exist
os.makedirs(captured_images_folder, exist_ok=True)

def capture_images(person_name):
    # Create a folder for the person's images
    person_folder = os.path.join(captured_images_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    print(f"Capturing images for {person_name}.")
    print("Please move your face slightly to different angles.")

    capture = cv2.VideoCapture(0)

    for i in range(50):
        ret, frame = capture.read()

        # Save the captured face image
        image_path = os.path.join(person_folder, f"captured_image_{i}.jpg")
        cv2.imwrite(image_path, frame)

        # Display a preview of the captured image
        cv2.imshow('Capture Images', frame)

        # Wait for a short delay (e.g., 200 milliseconds) between captures
        cv2.waitKey(200)

    # Release the capture and close the window
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Ask for the person's name
    person_name = input("Enter the person's name: ")
    capture_images(person_name)
