import cv2
import numpy as np
import tensorflow as tf
import torch

# Load YOLOv5 model (pretrained)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the pre-trained Keras model for gender classification
model_path = r'C:\Users\Hp\OneDrive\Desktop\Woman_Safety\models\Gender Prediction Model.h5'
gender_model = tf.keras.models.load_model(model_path)

# Preprocess image function for gender classification
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to classify gender
def classify_gender(image):
    processed_image = preprocess_image(image)
    prediction = gender_model.predict(processed_image)
    
    if prediction.shape[1] == 2:
        male_prob, female_prob = prediction[0]
        gender_label = 'Male' if male_prob > female_prob else 'Female'
    else:
        gender_label = 'Male' if prediction[0][0] > 0.5 else 'Female'
    
    return gender_label

# Function to calculate distance between two people
def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    distance = np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)
    return distance

# Function to process the video and detect proximity
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get screen size
    screen_width = 1920  # Adjust these values for your screen resolution
    screen_height = 1080

    # Create a window that resizes dynamically
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Calculate the scaling factor to fit the video inside the screen
        scale_width = screen_width / frame_width
        scale_height = screen_height / frame_height
        scale = min(scale_width, scale_height)

        # Resize the frame to fit the screen
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

        # Detect people in the frame using YOLO
        results = yolo_model(frame_resized)
        detections = results.pandas().xyxy[0]

        people_boxes = []
        genders = []
        
        for _, detection in detections.iterrows():
            if detection['name'] == 'person':
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                person_image = frame_resized[y1:y2, x1:x2]
                
                gender_prediction = classify_gender(person_image)
                people_boxes.append((x1, y1, x2, y2))
                genders.append(gender_prediction)

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f'Gender: {gender_prediction}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check for proximity between men and women
        assault_detected = False
        for i in range(len(people_boxes)):
            for j in range(i + 1, len(people_boxes)):
                if genders[i] != genders[j]:  # Checking only between opposite genders
                    distance = calculate_distance(people_boxes[i], people_boxes[j])
                    if distance < 500:  # Reduced threshold for sensitivity
                        assault_detected = True
                        break

        # Display "ASSAULT DETECTED!" if proximity threshold is crossed
        if assault_detected:
            cv2.putText(frame_resized, 'ASSAULT DETECTED!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Show the resized frame
        cv2.imshow('Video', frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing
if __name__ == "__main__":
    video_path = r'C:\Users\Hp\OneDrive\Desktop\Woman_Safety\videos\gettyimages-83361217-640_adpp.mp4'
    process_video(video_path)
    