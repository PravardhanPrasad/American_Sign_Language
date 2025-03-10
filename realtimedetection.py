import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("signlanguagedetectionmodel48x48.h5")

labels = ['A', 'M', 'N', 'S', 'T', 'blank']

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))             # Resize to 48x48
    image = np.expand_dims(image, axis=-1)          # Add channel dimension (HxWx1)
    image = np.expand_dims(image, axis=0)           # Add batch dimension (1x48x48x1)
    return image / 255.0                            # Normalize pixel values

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 0, 40, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    processed_roi = preprocess_image(roi)

    # Visualize the preprocessed ROI for debugging
    cv2.imshow("Processed ROI", processed_roi[0, :, :, 0])

    # Make prediction
    predictions = model.predict(processed_roi)
    predicted_label_index = np.argmax(predictions)
    predicted_label = labels[predicted_label_index]
    confidence = predictions[0][predicted_label_index] * 100

    # Display the prediction and confidence
    cv2.rectangle(frame, (x1, 0), (x2, y1), (0, 165, 255), -1)  # Top rectangle for text display
    if confidence > 50:  
        text = f"{predicted_label} {confidence:.2f}%"
    else:
        text = "Uncertain"
    cv2.putText(frame, text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw a rectangle around ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

    # Show the webcam output
    cv2.imshow("Sign Language Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
