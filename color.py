# face color analysis given eye center position

import sys
import os
import numpy as np
import cv2
import argparse
import time
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='sample/2.jpg', help="it can be image or video or webcan id")
parser.add_argument('--input_type', default='image', help= "either image or video (for video file and webcam id)")
opt = parser.parse_args()

# define HSV color ranges for eyes colors
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0] : ((121, 21, 50), (240, 100, 85)),
    class_name[1] : ((121, 2, 25), (300, 50, 75)),
    class_name[2] : ((5, 45, 20), (40, 100, 85)),
    class_name[3] : ((20, 3, 30), (65, 60, 60)),
    class_name[4] : ((0, 10, 5), (40, 40, 25)),
    class_name[5] : ((80, 21, 50), (120, 100, 85)),
    class_name[6] : ((80, 2, 25), (120, 50, 75))
}

def check_color(hsv, color):
    # Print HSV value being checked and the range it's being compared against
    print(f"Checking HSV: {hsv} against color range: {color}")
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and \
       (hsv[1] >= color[0][1]) and (hsv[1] <= color[1][1]) and \
       (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False


def find_class(hsv):
    color_id = 7
    print(f"Finding class for HSV: {hsv}")
    for i in range(len(class_name) - 1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            print(f"Matched class: {class_name[i]}")
            color_id = i
            break
    if color_id == 7:
        print("No match found, classified as 'Other'")
    return color_id

def eye_color(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    
    # Detect face
    result = detector.detect_faces(image)
    if not result:
        print('Warning: Cannot detect any face in the input image!')
        return
    
    bounding_box = result[0]['box']
    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    eye_radius = int(eye_distance / 17)  # approximate radius
    
    print(f"Bounding Box: {bounding_box}")
    print(f"Left Eye: {left_eye}, Right Eye: {right_eye}")
    print(f"Eye Distance: {eye_distance}, Eye Radius: {eye_radius}")
    
    # Define regions for left and right eyes
    def get_eye_region(center, radius):
        x_min = max(center[0] - radius, 0)
        x_max = min(center[0] + radius, w)
        y_min = max(center[1] - radius, 0)
        y_max = min(center[1] + radius, h)
        return x_min, x_max, y_min, y_max
    
    # Get bounding boxes for eyes
    left_eye_region = get_eye_region(left_eye, eye_radius)
    right_eye_region = get_eye_region(right_eye, eye_radius)
    
    # Analyze eye region pixels with sampling
    eye_class = np.zeros(len(class_name), dtype=float)
    pixel_count = 0  # Initialize pixel counter
    
    def analyze_region(region):
        nonlocal pixel_count  # Allow modifying the outer scope variable
        x_min, x_max, y_min, y_max = region
        sampling_step = 1  # Process every 2nd pixel for optimization
        for y in range(y_min, y_max, sampling_step):
            for x in range(x_min, x_max, sampling_step):
                # Check if pixel is inside the circular eye mask
                if (x - left_eye[0])**2 + (y - left_eye[1])**2 <= eye_radius**2 or \
                   (x - right_eye[0])**2 + (y - right_eye[1])**2 <= eye_radius**2:
                    hsv = imgHSV[y, x]
                    eye_class[find_class(hsv)] += 1
                    pixel_count += 1  # Increment pixel counter
    
    # Process left and right eye regions
    analyze_region(left_eye_region)
    analyze_region(right_eye_region)
    
    # Find dominant color
    main_color_index = np.argmax(eye_class[:-1])
    total_vote = eye_class.sum()
    
    print("\n\n--- Results ---")
    print(f"Total Votes: {total_vote}")
    print(f"Total Pixels Counted: {pixel_count}")  # Print total pixels analyzed
    for i, color in enumerate(class_name):
        print(f"{color}: {round(eye_class[i] / total_vote * 100, 2)}%")
    print(f"\nDominant Eye Color: {class_name[main_color_index]}")
    
    # Draw results on the image
    cv2.circle(image, left_eye, eye_radius, (0, 155, 255), 1)
    cv2.circle(image, right_eye, eye_radius, (0, 155, 255), 1)
    label = f'Dominant Eye Color: {class_name[main_color_index]}'
    cv2.putText(image, label, (left_eye[0] - 10, left_eye[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 255, 0))
    display_scale = 0.25  # Scale factor (50% of original size)
    resized_image = cv2.resize(image, (int(w * display_scale), int(h * display_scale)))
    
    # Display the resized image
    cv2.imshow('EYE-COLOR-DETECTION', resized_image)



if __name__ == '__main__':

    # image 
    if opt.input_type == 'image':   
        image = cv2.imread(opt.input_path, cv2.IMREAD_COLOR)
        # detect color percentage
        eye_color(image)
        cv2.imwrite('sample/result.jpg', image)    
        cv2.waitKey(0)

    # video or webcam
    else: 
        cap = cv2.VideoCapture(opt.input_path)
        while(True):
            ret, frame = cap.read()
            if ret == -1: 
                break

            eye_color(frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
