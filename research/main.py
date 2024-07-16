import os
import cv2
import imutils
import numpy as np
from deepface import DeepFace

# Paths to the face detector model files
prototxtPath = r"model\deploy.prototxt"
weightsPath = r"model\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Path to the image file
image_path = r'test.jpg'  # Replace with your image file path

# Check if the image file exists
if not os.path.exists(image_path):
    print("Error: Image file does not exist.")
    exit()

# Load the image
image = cv2.imread(image_path)
frame = imutils.resize(image, width=800)

(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

faceNet.setInput(blob)
detections = faceNet.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        # Compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Ensure the bounding boxes fall within the dimensions of the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        try:
            result = DeepFace.analyze(frame, actions=['emotion'])
            color = (0, 0, 255)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, "{}".format(result['dominant_emotion']), (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        except Exception as e:
            print(e)

# Save the processed image
output_path = r'processed_test.jpg'  # Path to save the processed image
cv2.imwrite(output_path, frame)

# Display the processed image
cv2.imshow("Processed Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
