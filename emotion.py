import cv2
from deepface import DeepFace

# Load the face cascade classifier (pre-trained model for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video using your webcam (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame capture was successful
    if not ret:
        print("Error: Unable to capture frame from webcam.")
        break

    # Convert the frame to grayscale for face detection efficiency
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the cascade classifier
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI) from the original RGB frame
        face_roi = frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI using DeepFace
        try:
            # Analyze emotions, enforcing face detection within DeepFace
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=True)
        except:
            # Handle potential exceptions during analysis (e.g., network errors)
            print("Error: Emotion analysis failed.")
            continue

        # Get the emotion probabilities
        emotion_probs = result[0]['emotion']

        # Ensure probabilities sum to 1 (handle potential errors)
        total_prob = sum(emotion_probs.values())
        if total_prob > 0:
            for emotion in emotion_probs:
                emotion_probs[emotion] /= total_prob  # Normalize probabilities

        # Determine the dominant emotion (emotion with the highest probability)
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)

        # Calculate the percentage of the dominant emotion, clamped to 100%
        percentage = min(emotion_probs[dominant_emotion] * 100, 100)  # Clamp to 100

        # Format the percentage with two decimal places
        percentage_text = f"{percentage:.2f}%"

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the dominant emotion and its percentage on the frame
        cv2.putText(frame, f"{dominant_emotion}: {percentage_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame with detected faces and emotion labels
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
