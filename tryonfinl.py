import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Set the frame dimensionsq
frame_width = 1280
frame_height = 1000
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Create a window for the video feed
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", frame_width, frame_height)

# Initialize the Pose Detector
detector = PoseDetector()

# Path to the shirt images
shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
print(listShirts)

# Define the scale factor to resize the shirt
shirt_scale = 0.7  # Adjust this value to resize the shirt

# Load button images
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)

# Initialize variables for shirt selection
imageNumber = 0
selectionSpeed = 10
counterRight = 0
counterLeft = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Find pose and position
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
    
    if lmList:
        # Load and resize the shirt image
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        imgShirt_resized = cv2.resize(imgShirt, None, fx=shirt_scale, fy=shirt_scale)
        shirt_width = imgShirt_resized.shape[1]
        center_x = (frame_width - shirt_width) // 2

        # Overlay the resized shirt image
        img = cvzone.overlayPNG(img, imgShirt_resized, (center_x, 100))

 # Overlay buttons on the screen
        if imgButtonRight is not None:
            img = cvzone.overlayPNG(img, imgButtonRight, (72, 293))
        if imgButtonLeft is not None:
            img = cvzone.overlayPNG(img, imgButtonLeft, (1074, 293))

        # Check for hand gestures to change shirts
        if lmList[16][1] < 300:
            # If right hand is raised
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                imageNumber = (imageNumber + 1) % len(listShirts)  # Loop to the beginning if at the end
        elif lmList[15][1] < 300:
            # If left hand is raised
            counterLeft += 1
            cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                imageNumber = (imageNumber - 1) % len(listShirts)  # Loop to the end if at the beginning
        else:
            counterRight = 0
            counterLeft = 0

    # Display the image
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
