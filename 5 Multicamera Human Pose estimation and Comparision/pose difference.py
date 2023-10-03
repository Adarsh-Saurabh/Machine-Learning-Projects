import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_matching_percentage(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return 0
    num_landmarks = min(len(landmarks1.landmark), len(landmarks2.landmark))
    total_distance = 0
    for i in range(num_landmarks):
        landmark1 = landmarks1.landmark[i]
        landmark2 = landmarks2.landmark[i]
        distance = ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5
        total_distance += distance
    matching_percentage = 100 - (total_distance / num_landmarks * 100)
    return matching_percentage

cap1 = cv2.VideoCapture('child jumping 2.mp4') # put the newbie video here

cap2 = cv2.VideoCapture('child jumping 3.mp4') # put the expert video here

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap1.isOpened() and cap2.isOpened():
        # time.sleep(1)
        success1, image1 = cap1.read()
        success2, image2 = cap2.read()
        if not success1 or not success2:
            print("No video in camera frame")
            break

        image1 = cv2.flip(image1, 1) # mirror the image horizontally
        h1, w1, c1 = image1.shape
        fps_start_time1 = time.time()
        fps1 = 0

        image2 = cv2.flip(image2, 1) # mirror the image horizontally
        h2, w2, c2 = image2.shape
        fps_start_time2 = time.time()
        fps2 = 0

        image1.flags.writeable = False
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        result1 = pose.process(image1)

        image1.flags.writeable = True
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image1,
            result1.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        fps_end_time1 = time.time()
        fps1 = 1 / (fps_end_time1 - fps_start_time1) # calculate the FPS
        cv2.putText(image1, f"FPS: {int(fps1)//2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # cv2.imshow("pose estimation 1", image1)

        image2.flags.writeable = False
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        result2 = pose.process(image2)

        image2.flags.writeable = True
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image2,
            result2.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        fps_end_time2 = time.time()
        fps2 = 1 / (fps_end_time2 - fps_start_time2) # calculate the FPS
        cv2.putText(image2, f"FPS: {int(fps1)//2}"    , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(image2, 'Expert'    , (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("pose estimation 2", image2)

        matching_percentage = calculate_matching_percentage(result1.pose_landmarks, result2.pose_landmarks)
        print(f"Matching Percentage: {matching_percentage}%")
        cv2.putText(image1, f"Match : {int(matching_percentage) }"    , (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image1, 'Newbie'    , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("pose estimation 1", image1)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap1.release()
cap2.release()
cv2.destroyAllWindows()

