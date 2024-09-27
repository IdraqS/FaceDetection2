import cv2
import tensorflow as tf
import numpy as np

MODEL_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\trained_models\trained_models_final\model_11_4.keras'
CASCADES_DIR = r'C:\Users\Idraq\Desktop\Project\Python Files\haarcascade_frontalface_default.xml'

#load model and haar cascades classifier
model = tf.keras.models.load_model(MODEL_DIR)
cascades = cv2.CascadeClassifier(CASCADES_DIR)

if model is None:
    raise Exception ("WHERE IS MY MODEL?!")

# Initialise video capture
webcam = cv2.VideoCapture(0) 

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Convert frame to grayscale for Haar cascade
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haarcascades to detect face and return values x,y,w,h (of the faces) in an array.
    grey_faces = cascades.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 4, minSize = (40, 40))
    
    for (x, y, w, h) in grey_faces:
        
        #resize faces to (75,75) to fit in model using xywh values
        grey_faces_resized = cv2.resize(grey[y:y+h, x:x+w], (75, 75))
        
        # normalise face and add batch dimension to process 1 image at time
        normalised_faces = grey_faces_resized / 255.0
        face_input = np.expand_dims(normalised_faces, axis = (0, -1))
        
        # Make model predict me or not me
        prediction = model.predict(face_input)
        prediction_score = prediction[0][0] 
        prediction_percentage = prediction_score * 100 # pred as % to show in on label
        threshold_value = 0.85 #threshold high bc binary classification + small dataset
    
        if prediction_score > threshold_value: 
            label = f'Idraq: {prediction_percentage:.2f}%'
            color = (0, 255, 0)  # Green if me
        else:
            label = f'Not Idraq: {100 - prediction_percentage:.2f}'
            color = (0, 0, 255)  # Red if not me!!
        
        # Draw rectangle around face and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the result
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()