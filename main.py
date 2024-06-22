import numpy as np
import cv2
import mediapipe as mp
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Mediapipe drawing class, useful for annotation
mp_drawing = mp.solutions.drawing_utils

# Variable to store the optimal distance when sitting up straight
optimal_shoulder_height = None
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.music.load('alarm.wav')
        pygame.mixer.music.play(-1)  # Play in a loop

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

def is_slouching(landmarks, optimal_dist):
    if optimal_dist is None:
        return False

    # Get the coordinates for shoulders and nose
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    # Calculate the average shoulder height
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Calculate vertical distance of nose from average shoulder height
    # nose_to_shoulder_dist = avg_shoulder_y - nose.y

    print(f"Current distance: {avg_shoulder_y}, Optimal distance: {optimal_dist}")
    # Determine slouching based on the vertical distance threshold
    return avg_shoulder_y * 0.99 > optimal_dist
    # return nose_to_shoulder_dist < optimal_dist

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect the pose
    result = pose.process(rgb_frame)
    
    # Draw the pose annotation on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if optimal_shoulder_height is None:
            cv2.putText(frame, "Sit up straight and press Space to set optimal position", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Check if the human is slouching
            if is_slouching(result.pose_landmarks.landmark, optimal_shoulder_height):
                cv2.putText(frame, "Slouching detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                play_alarm()
            else:
                stop_alarm()
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Capture the optimal distance when space is pressed
        if result.pose_landmarks:
            left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            print(left_shoulder.y)
            print(right_shoulder.y)
            optimal_shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
            print(f"Optimal nose to shoulder distance set: {optimal_shoulder_height}")
        # if result.pose_landmarks:
        #     left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        #     right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        #     nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
        #     avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        #     optimal_shoulder_height = avg_shoulder_y - nose.y
        #     print(f"Optimal nose to shoulder distance set: {optimal_shoulder_height}")

# Release the capture
cap.release()
cv2.destroyAllWindows()
