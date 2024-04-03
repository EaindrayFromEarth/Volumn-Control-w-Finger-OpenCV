import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to adjust volume based on distance
def adjust_volume(distance):
    # Normalize distance to a range of 0 to 1
    normalized_distance = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    # Calculate volume level based on normalized distance
    volume_level = int(normalized_distance * 100)
    # Adjust laptop volume using pyautogui
    pyautogui.press('volumedown') if volume_level < 50 else pyautogui.press('volumeup')

# Main function to capture webcam feed and perform volume control
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image with Mediapipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Get coordinates of thumb and index finger
                thumb_coords = (int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0]))
                index_finger_coords = (int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0]))

                # Calculate distance between thumb and index finger
                distance = calculate_distance(thumb_coords, index_finger_coords)

                # Adjust volume based on distance
                adjust_volume(distance)

                # Draw lines connecting thumb and index finger for visualization
                cv2.line(frame, thumb_coords, index_finger_coords, (0, 255, 0), 2)

        cv2.imshow('Hand Volume Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MIN_DISTANCE = 50  # Minimum distance between thumb and index finger
    MAX_DISTANCE = 300  # Maximum distance between thumb and index finger
    main()
