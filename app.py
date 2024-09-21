from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, jsonify
import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder path and allowed extensions
UPLOAD_FOLDER = r'C:\Users\Hp\OneDrive\Desktop\Woman_Safety\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Global list to store assault locations
assault_locations = []

# Load models
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
gender_model = tf.keras.models.load_model('models/Gender Prediction Model.h5')

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process uploaded video and run assault detection
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get the original frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    result_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create VideoWriter object with original dimensions
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 person detection
        results = yolo_model(frame)
        detections = results.pandas().xyxy[0]

        people_boxes = []
        genders = []

        for _, detection in detections.iterrows():
            if detection['name'] == 'person':
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                
                if x1 < x2 and y1 < y2:
                    person_image = frame[y1:y2, x1:x2]
                    
                    # Ensure the person image is valid
                    if person_image.size == 0:
                        continue
                    
                    # Gender classification
                    processed_image = cv2.resize(person_image, (224, 224)) / 255.0
                    processed_image = np.expand_dims(processed_image, axis=0)
                    prediction = gender_model.predict(processed_image)
                    male_prob, female_prob = prediction[0]
                    gender = 'Male' if male_prob > female_prob else 'Female'
                    
                    people_boxes.append((x1, y1, x2, y2))
                    genders.append(gender)
                    
                    # Draw the bounding boxes and gender label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check proximity between men and women
        assault_detected = False
        for i in range(len(people_boxes)):
            for j in range(i + 1, len(people_boxes)):
                if genders[i] != genders[j]:
                    # Calculate distance between people
                    x1_center = (people_boxes[i][0] + people_boxes[i][2]) / 2
                    y1_center = (people_boxes[i][1] + people_boxes[i][3]) / 2
                    x2_center = (people_boxes[j][0] + people_boxes[j][2]) / 2
                    y2_center = (people_boxes[j][1] + people_boxes[j][3]) / 2
                    distance = np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)
                    if distance < 500:  # Set sensitivity for assault detection
                        assault_detected = True
                        break

        if assault_detected:
            cv2.putText(frame, 'ASSAULT DETECTED!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            # Store the assault location (latitude, longitude)
            assault_locations.append({'lat': 28.544, 'lng': 77.5454})  # Replace with actual detection coordinates

        # Write the frame into the output video
        out.write(frame)

    cap.release()
    out.release()
    return result_video_path

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # Process the uploaded video
        result_video_path = process_video(video_path)
        
        # Provide download link for processed video
        return redirect(url_for('download_file', filename='output_video.mp4'))

# Route to download processed video
@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# Route for live camera feed
@app.route('/live_feed')
def live_camera_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 person detection
        results = yolo_model(frame)
        detections = results.pandas().xyxy[0]

        people_boxes = []
        genders = []

        for _, detection in detections.iterrows():
            if detection['name'] == 'person':
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x1 < x2 and y1 < y2:
                    person_image = frame[y1:y2, x1:x2]
                    
                    # Ensure the person image is valid
                    if person_image.size == 0:
                        continue
                    
                    # Gender classification
                    processed_image = cv2.resize(person_image, (224, 224)) / 255.0
                    processed_image = np.expand_dims(processed_image, axis=0)
                    prediction = gender_model.predict(processed_image)
                    male_prob, female_prob = prediction[0]
                    gender = 'Male' if male_prob > female_prob else 'Female'
                    
                    people_boxes.append((x1, y1, x2, y2))
                    genders.append(gender)
                    
                    # Draw the bounding boxes and gender label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check proximity between men and women
        assault_detected = False
        for i in range(len(people_boxes)):
            for j in range(i + 1, len(people_boxes)):
                if genders[i] != genders[j]:
                    # Calculate distance between people
                    x1_center = (people_boxes[i][0] + people_boxes[i][2]) / 2
                    y1_center = (people_boxes[i][1] + people_boxes[i][3]) / 2
                    x2_center = (people_boxes[j][0] + people_boxes[j][2]) / 2
                    y2_center = (people_boxes[j][1] + people_boxes[j][3]) / 2
                    distance = np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)
                    if distance < 500:  # Set sensitivity for assault detection
                        assault_detected = True
                        break

        if assault_detected:
            cv2.putText(frame, 'ASSAULT DETECTED!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            # Store the assault location (latitude, longitude)
            assault_locations.append({'lat': 28.544, 'lng': 77.5454})  # Replace with actual detection coordinates

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to get assault locations
@app.route('/assault_locations', methods=['GET'])
def get_assault_locations():
    return jsonify(assault_locations)

if __name__ == '__main__':
    app.run(debug=True)