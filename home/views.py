from datetime import date
from decimal import Decimal

from django.http import HttpResponse
from django.shortcuts import render
import math
import cv2
import mediapipe as mp
import time
import numpy as np
from .models import RegisteredUser, AasnaVideo, SquatRecord, FitnessProfile, CurlRecord, PushupRecord, YogaAsanaRecord
from keras.models import load_model


def index(request):
    return render(request,'index.html',{'incorrect':'Login and Sign Up'})

def signup(request):
    name = request.POST.get('name')
    age = request.POST.get('age')
    gender = request.POST.get('gender')
    uname = request.POST.get('uname')
    passw = request.POST.get('passw')
    weight = request.POST.get('weight')
    RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
    if RegisteredUser1 != None:
        return render(request,'index.html',{'incorrect':'Username already exists'})
    else:
        RegisteredUser1 = RegisteredUser.objects.create(full_name=name,age=age,gender=gender,username=uname,password=passw,weight=weight)
        RegisteredUser1.save()
        return render(request,'index.html',{'incorrect':'Registered Successfully'})

def login(request):
    uname = request.POST.get('uname')
    passw = request.POST.get('passw')
    RegisteredUser1 = RegisteredUser.objects.filter(username=uname , password=passw).first()
    ucount = RegisteredUser.objects.count()
    AasnaVideo1 = AasnaVideo.objects.all()

    today = date.today()
    if RegisteredUser1 != None:
        CurlRecord1 = CurlRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if CurlRecord1:
            pass
        else:
            CurlRecord1 = CurlRecord.objects.create(user=RegisteredUser1, date=today)

        SquatRecord1 = SquatRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if SquatRecord1:
            pass
        else:
            SquatRecord1 = SquatRecord.objects.create(user=RegisteredUser1, date=today)

        PushupRecord1 = PushupRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if PushupRecord1:
            pass
        else:
            PushupRecord1 = PushupRecord.objects.create(user=RegisteredUser1, date=today)

        FitnessProfile1 = FitnessProfile.objects.filter(user=RegisteredUser1, date=today).first()
        if FitnessProfile1:
            pass
        else:
            FitnessProfile1 = FitnessProfile.objects.create(user=RegisteredUser1, date=today)
            FitnessProfile1.save()

        caloriesB = FitnessProfile1.calculate_calories_burned()

        curlCal = caloriesB[0]
        squatCal = caloriesB[1]
        pushupCal = caloriesB[2]

        rem = FitnessProfile1.target_calories - FitnessProfile1.calories_burnt
        weight_kg = RegisteredUser1.weight
        curlbr = (Decimal('2.8') * weight_kg * Decimal('3.5')) / Decimal('100')
        squatbr = (Decimal('3.5') * weight_kg * Decimal('3.5')) / Decimal('100')
        pushupbr = (Decimal('4.4') * weight_kg * Decimal('3.5')) / Decimal('100')

        YogaAsanaRecord1 = YogaAsanaRecord.objects.filter(fitness_profile=FitnessProfile1,date=today)
        context = {
            'ucount': ucount,
            'aasna': AasnaVideo1,
            'reg': RegisteredUser1,
            'fit': FitnessProfile1,
            'curls': CurlRecord1,
            'squats': SquatRecord1,
            'pushup': PushupRecord1,
            'curlCal': curlCal,
            'squatCal': squatCal,
            'pushupCal': pushupCal,
            'rem': rem,
            'curlbr': curlbr,
            'squatbr': squatbr,
            'pushupbr': pushupbr,
            'yog':YogaAsanaRecord1,
        }
        return render(request,'dashboard.html',context)

    else:
        return render(request,'index.html',{'incorrect':'Incorrect Username or Password'})

class poseDetector():
    def __init__(self , mode= False , upBody=False , smooth=True, detectionCon= 0.5 , trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, True, True)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)  # Store the results in self.results
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self,img,draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id , lm in enumerate(self.results.pose_landmarks.landmark):
                h , w , c = img.shape
                cx , cy = int(lm.x * w ), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img , (cx,cy), 5 , (255,0,0) , cv2.FILLED)
        return self.lmList

    def findAngle(self,img , p1,p2,p3 , draw=True):

        #Get the landmarks
        x1 , y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-
                                        math.atan2(y1-y2,x1-x2))
        if angle<0:
            angle+=360

        #Draw
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img,str(int(angle)),(x2-50,y2+50),
                        cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

        return angle

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)
        if len(lmList) != 0 :
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 250), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow("Image",img)
        key = cv2.waitKey(1)
        if key==ord('q'):
            break

def curlsCounter(request,uname):
    cap = cv2.VideoCapture(0)
    detector = poseDetector()
    count = 0;
    dir = 0
    color = (0, 120, 255)
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1380, 780))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 12, 14, 16)
            color = (0, 120, 255)

            # Left Arm
            angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (210, 310), (650, 100))
            # Check for the dumbbell curls
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            print(per)
            print(count)

            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

            # Draw Curl Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (50, 650), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 5)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
            today = date.today()
            CurlRecord1 = CurlRecord.objects.filter(user=RegisteredUser1, date=today).first()
            if CurlRecord1:
                CurlRecord1.curls_done += count
                CurlRecord1.save()
            else:
                CurlRecord1 = CurlRecord.objects.create(user=RegisteredUser1, date=today,curls_done=count)
            break
    cv2.destroyAllWindows()

    RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
    ucount = RegisteredUser.objects.count()
    AasnaVideo1 = AasnaVideo.objects.all()

    today = date.today()

    if RegisteredUser1 != None:
        CurlRecord1 = CurlRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if CurlRecord1:
            pass
        else:
            CurlRecord1 = CurlRecord.objects.create(user=RegisteredUser1, date=today)

        SquatRecord1 = SquatRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if SquatRecord1:
            pass
        else:
            SquatRecord1 = SquatRecord.objects.create(user=RegisteredUser1, date=today)

        PushupRecord1 = PushupRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if PushupRecord1:
            pass
        else:
            PushupRecord1 = PushupRecord.objects.create(user=RegisteredUser1, date=today)

        FitnessProfile1 = FitnessProfile.objects.filter(user=RegisteredUser1, date=today).first()
        if FitnessProfile1:
            pass
        else:
            FitnessProfile1 = FitnessProfile.objects.create(user=RegisteredUser1, date=today)
            FitnessProfile1.save()

        caloriesB = FitnessProfile1.calculate_calories_burned()

        curlCal = caloriesB[0]
        squatCal = caloriesB[1]
        pushupCal = caloriesB[2]

        rem = FitnessProfile1.target_calories - FitnessProfile1.calories_burnt
        weight_kg = RegisteredUser1.weight
        curlbr = (Decimal('2.8') * weight_kg * Decimal('3.5')) / Decimal('100')
        squatbr = (Decimal('3.5') * weight_kg * Decimal('3.5')) / Decimal('100')
        pushupbr = (Decimal('4.4') * weight_kg * Decimal('3.5')) / Decimal('100')

        YogaAsanaRecord1 = YogaAsanaRecord.objects.filter(fitness_profile=FitnessProfile1, date=today)
        context = {
            'ucount': ucount,
            'aasna': AasnaVideo1,
            'reg': RegisteredUser1,
            'fit': FitnessProfile1,
            'curls': CurlRecord1,
            'squats': SquatRecord1,
            'pushup': PushupRecord1,
            'curlCal': curlCal,
            'squatCal': squatCal,
            'pushupCal': pushupCal,
            'rem': rem,
            'curlbr': curlbr,
            'squatbr': squatbr,
            'pushupbr': pushupbr,
            'yog': YogaAsanaRecord1,
        }
    return render(request, 'dashboard.html',context)

def squatsCounter(request,uname):
    def findAngle(a, b, c, minVis=0.8):
        # Finds the angle at b with endpoints a and c
        # Returns -1 if below minimum visibility threshold
        # Takes lm_arr elements

        if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
            bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
            ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])

            angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba)
                                                  * np.linalg.norm(bc))) * (180 / np.pi)

            if angle > 180:
                return 360 - angle
            else:
                return angle
        else:
            return -1

    def legState(angle):
        if angle < 0:
            return 0  # Joint is not being picked up
        elif angle < 105:
            return 1  # Squat range
        elif angle < 150:
            return 2  # Transition range
        else:
            return 3  # Upright range


    # Init mediapipe drawing and pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

        # Init Video Feed
        # Opens file if passed as parameter from terminal
        # Else Defaults to webcam

    cap = None
    cap = cv2.VideoCapture(0)

        # Main Detection Loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            # Initialize Reps and Body State
            repCount = 0
            lastState = 9

            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    print('Error: Image not found or could not be loaded.')
                else:
                    frame = cv2.resize(frame, (1024, 600))

                # frame = cv2.resize(frame, (1280, 800),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

                if ret == True:
                    try:
                        # Convert frame to RGB
                        # Writeable = False forces pass by ref (faster)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame.flags.writeable = False

                        # Detect Pose Landmarks
                        # lm used for drawing
                        # lm_arr is actually indexable with .x, .y, .z attr
                        lm = pose.process(frame).pose_landmarks
                        lm_arr = lm.landmark
                    except:
                        print("Please Step Into Frame")
                        cv2.imshow("Squat Rep Counter", frame)
                        cv2.waitKey(1)
                        continue

                    # Allow write, convert back to BGR
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Draw overlay with parameters:
                    # (frame, landmarks, list of connected landmarks, landmark draw spec, connection draw spec)
                    mp_drawing.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(
                        0, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                    # Calculate Angle
                    # Hip -Knee-Foot Indices:
                    # R: 24, 26, 28
                    # L: 23, 25, 27
                    rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
                    lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])

                    # Calculate state
                    rState = legState(rAngle)
                    lState = legState(lAngle)
                    state = rState * lState

                    # Final state is product of two leg states
                    # 0 -> One or both legs not being picked up
                    # Even -> One or both legs are still transitioning
                    # Odd
                    #   1 -> Squatting
                    #   9 -> Upright
                    #   3 -> One squatting, one upright

                    # Only update lastState on 1 or 9

                    if state == 0:  # One or both legs not detected
                        if rState == 0:
                            print("Right Leg Not Detected")
                        if lState == 0:
                            print("Left Leg Not Detected")
                    elif state % 2 == 0 or rState != lState:  # One or both legs still transitioning
                        if lastState == 1:
                            if lState == 2 or lState == 1:
                                print("Fully extend left leg")
                            if rState == 2 or lState == 1:
                                print("Fully extend right leg")
                        else:
                            if lState == 2 or lState == 3:
                                print("Fully retract left leg")
                            if rState == 2 or lState == 3:
                                print("Fully retract right leg")
                    else:
                        if state == 1 or state == 9:
                            if lastState != state:
                                lastState = state
                                if lastState == 1:
                                    print("GOOD!")
                                    # for playing note.wav file
                                    repCount += 1
                    # Display squat count on the image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (50, 50)
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    image = frame
                    image = cv2.putText(image, "Squat Count: " + str(repCount), org, font, fontScale, color,
                                        thickness, cv2.LINE_AA)

                    print("Squats: " + (str)(repCount))
                    image = cv2.putText(image, "Squat Count: " + str(repCount), org, font, fontScale, color, thickness,
                                        cv2.LINE_AA)

                    cv2.imshow("Squat Rep Counter", frame)
                    if cv2.waitKey(1) == ord('q'):
                        RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
                        today = date.today()
                        SquatRecord1 = SquatRecord.objects.filter(user=RegisteredUser1, date=today).first()
                        if SquatRecord1:
                            SquatRecord1.squats_done += repCount
                            SquatRecord1.save()
                        else:
                            SquatRecord1 = SquatRecord.objects.create(user=RegisteredUser1, date=today,
                                                                        squats_done=repCount)
                        break
    cv2.destroyAllWindows()
    RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
    ucount = RegisteredUser.objects.count()
    AasnaVideo1 = AasnaVideo.objects.all()

    today = date.today()

    if RegisteredUser1 != None:
        CurlRecord1 = CurlRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if CurlRecord1:
            pass
        else:
            CurlRecord1 = CurlRecord.objects.create(user=RegisteredUser1, date=today)

        SquatRecord1 = SquatRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if SquatRecord1:
            pass
        else:
            SquatRecord1 = SquatRecord.objects.create(user=RegisteredUser1, date=today)

        PushupRecord1 = PushupRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if PushupRecord1:
            pass
        else:
            PushupRecord1 = PushupRecord.objects.create(user=RegisteredUser1, date=today)

        FitnessProfile1 = FitnessProfile.objects.filter(user=RegisteredUser1, date=today).first()
        if FitnessProfile1:
            pass
        else:
            FitnessProfile1 = FitnessProfile.objects.create(user=RegisteredUser1, date=today)
            FitnessProfile1.save()

        caloriesB = FitnessProfile1.calculate_calories_burned()

        curlCal = caloriesB[0]
        squatCal = caloriesB[1]
        pushupCal = caloriesB[2]

        rem = FitnessProfile1.target_calories - FitnessProfile1.calories_burnt
        weight_kg = RegisteredUser1.weight
        curlbr = (Decimal('2.8') * weight_kg * Decimal('3.5')) / Decimal('100')
        squatbr = (Decimal('3.5') * weight_kg * Decimal('3.5')) / Decimal('100')
        pushupbr = (Decimal('4.4') * weight_kg * Decimal('3.5')) / Decimal('100')

        YogaAsanaRecord1 = YogaAsanaRecord.objects.filter(fitness_profile=FitnessProfile1, date=today)
        context = {
            'ucount': ucount,
            'aasna': AasnaVideo1,
            'reg': RegisteredUser1,
            'fit': FitnessProfile1,
            'curls': CurlRecord1,
            'squats': SquatRecord1,
            'pushup': PushupRecord1,
            'curlCal': curlCal,
            'squatCal': squatCal,
            'pushupCal': pushupCal,
            'rem': rem,
            'curlbr': curlbr,
            'squatbr': squatbr,
            'pushupbr': pushupbr,
            'yog': YogaAsanaRecord1,
        }

    return render(request, 'dashboard.html', context)

def pushupCounter(request,uname):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    def distanceCalculate(p1, p2):
        """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    pushUpStart = 0
    pushUpCount = 0

    # For static images:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192)  # gray

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:

        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue

            print(f'Nose coordinates: ('
                  f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                  f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})')

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.get_default_pose_landmarks_style())  # Fix for mp_drawing_styles

            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # For webcam input:
    cap = cv2.VideoCapture(0)

    # Create a window with the name "MediaPipe Pose" and set it to almost full screen size
    cv2.namedWindow("MediaPipe Pose", 0)

    # Set the window width and height manually (adjust these values as needed)
    window_width = 1380
    window_height = 720
    cv2.resizeWindow("MediaPipe Pose", window_width, window_height)

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            results = pose.process(image)

            if results is not None and results.pose_landmarks is not None:
                image_height, image_width, _ = image.shape

                nosePoint = (int(results.pose_landmarks.landmark[0].x * image_width),
                             int(results.pose_landmarks.landmark[0].y * image_height))
                leftWrist = (int(results.pose_landmarks.landmark[15].x * image_width),
                             int(results.pose_landmarks.landmark[15].y * image_height))
                rightWrist = (int(results.pose_landmarks.landmark[16].x * image_width),
                              int(results.pose_landmarks.landmark[16].y * image_height))
                leftShoulder = (int(results.pose_landmarks.landmark[11].x * image_width),
                                int(results.pose_landmarks.landmark[11].y * image_height))
                rightShoulder = (int(results.pose_landmarks.landmark[12].x * image_width),
                                 int(results.pose_landmarks.landmark[12].y * image_height))

                # Push-up counting logic here
                if distanceCalculate(rightShoulder, rightWrist) < 130:
                    pushUpStart = 1
                elif pushUpStart and distanceCalculate(rightShoulder, rightWrist) > 250:
                    pushUpCount = pushUpCount + 1
                    pushUpStart = 0

                print(pushUpCount)

                font = cv2.FONT_HERSHEY_SIMPLEX
                # org
                org = (50, 100)
                # fontScale
                fontScale = 1.6
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 3

                image = cv2.putText(image, "Push-up count:  " + str(pushUpCount), org, font, fontScale, color,
                                    thickness, cv2.LINE_AA)

                # Draw pose landmarks on the image.
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 144, 30), thickness=2))

            else:
                # Handle the case when pose landmarks are not detected.
                print("Pose landmarks not detected.")

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) == ord('q'):
                break
            time.sleep(0.01)

    cap.release()

    cv2.destroyAllWindows()
    RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
    ucount = RegisteredUser.objects.count()
    AasnaVideo1 = AasnaVideo.objects.all()
    today = date.today()

    if RegisteredUser1 != None:
        CurlRecord1 = CurlRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if CurlRecord1:
            pass
        else:
            CurlRecord1 = CurlRecord.objects.create(user=RegisteredUser1, date=today)

        SquatRecord1 = SquatRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if SquatRecord1:
            pass
        else:
            SquatRecord1 = SquatRecord.objects.create(user=RegisteredUser1, date=today)

        PushupRecord1 = PushupRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if PushupRecord1:
            pass
        else:
            PushupRecord1 = PushupRecord.objects.create(user=RegisteredUser1, date=today)

        FitnessProfile1 = FitnessProfile.objects.filter(user=RegisteredUser1, date=today).first()
        if FitnessProfile1:
            pass
        else:
            FitnessProfile1 = FitnessProfile.objects.create(user=RegisteredUser1, date=today)
            FitnessProfile1.save()

        caloriesB = FitnessProfile1.calculate_calories_burned()

        curlCal = caloriesB[0]
        squatCal = caloriesB[1]
        pushupCal = caloriesB[2]

        rem = FitnessProfile1.target_calories - FitnessProfile1.calories_burnt
        weight_kg = RegisteredUser1.weight
        curlbr = (Decimal('2.8') * weight_kg * Decimal('3.5')) / Decimal('100')
        squatbr = (Decimal('3.5') * weight_kg * Decimal('3.5')) / Decimal('100')
        pushupbr = (Decimal('4.4') * weight_kg * Decimal('3.5')) / Decimal('100')

        YogaAsanaRecord1 = YogaAsanaRecord.objects.filter(fitness_profile=FitnessProfile1, date=today)
        context = {
            'ucount': ucount,
            'aasna': AasnaVideo1,
            'reg': RegisteredUser1,
            'fit': FitnessProfile1,
            'curls': CurlRecord1,
            'squats': SquatRecord1,
            'pushup': PushupRecord1,
            'curlCal': curlCal,
            'squatCal': squatCal,
            'pushupCal': pushupCal,
            'rem': rem,
            'curlbr': curlbr,
            'squatbr': squatbr,
            'pushupbr': pushupbr,
            'yog': YogaAsanaRecord1,
        }

    return render(request, 'dashboard.html', context)

def yogaTrainer(request,uname):
    def inFrame(lst):
        if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[
            16].visibility > 0.6:
            return True
        return False

    model = load_model("model.h5")
    label = np.load("labels.npy")

    holistic = mp.solutions.pose
    holis = holistic.Pose()
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    list1=[]
    while True:
        lst = []

        _, frm = cap.read()

        window = np.zeros((940, 940, 3), dtype="uint8")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        frm = cv2.blur(frm, (4, 4))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model.predict(lst)
            pred = label[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(window, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
                today = date.today()
                RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
                FitnessProfile1 = FitnessProfile.objects.filter(user=RegisteredUser1,date=today).first()
                YogaAsana1 = YogaAsanaRecord.objects.filter(fitness_profile=FitnessProfile1,aasna=pred).first()
                if YogaAsana1:
                    if YogaAsana1.aasna in list1:
                        pass
                    else:
                        YogaAsana1.times += 1
                        YogaAsana1.save()
                        list1.append(YogaAsana1.aasna)
                else :
                    YogaAsana1 = YogaAsanaRecord.objects.create(fitness_profile=FitnessProfile1, aasna=pred)
                helo = YogaAsana1.comp()
            else:
                cv2.putText(window, "Asana is either wrong not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255),
                            3)

        else:
            cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                               connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3,
                                                                         thickness=3))

        window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))

        cv2.imshow("window", window)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    RegisteredUser1 = RegisteredUser.objects.filter(username=uname).first()
    ucount = RegisteredUser.objects.count()
    AasnaVideo1 = AasnaVideo.objects.all()
    today = date.today()

    if RegisteredUser1 != None:
        CurlRecord1 = CurlRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if CurlRecord1:
            pass
        else:
            CurlRecord1 = CurlRecord.objects.create(user=RegisteredUser1, date=today)

        SquatRecord1 = SquatRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if SquatRecord1:
            pass
        else:
            SquatRecord1 = SquatRecord.objects.create(user=RegisteredUser1, date=today)

        PushupRecord1 = PushupRecord.objects.filter(user=RegisteredUser1, date=today).first()
        if PushupRecord1:
            pass
        else:
            PushupRecord1 = PushupRecord.objects.create(user=RegisteredUser1, date=today)

        FitnessProfile1 = FitnessProfile.objects.filter(user=RegisteredUser1, date=today).first()
        if FitnessProfile1:
            pass
        else:
            FitnessProfile1 = FitnessProfile.objects.create(user=RegisteredUser1, date=today)
            FitnessProfile1.save()

        caloriesB = FitnessProfile1.calculate_calories_burned()

        curlCal = caloriesB[0]
        squatCal = caloriesB[1]
        pushupCal = caloriesB[2]

        rem = FitnessProfile1.target_calories - FitnessProfile1.calories_burnt
        weight_kg = RegisteredUser1.weight
        curlbr = (Decimal('2.8') * weight_kg * Decimal('3.5')) / Decimal('100')
        squatbr = (Decimal('3.5') * weight_kg * Decimal('3.5')) / Decimal('100')
        pushupbr = (Decimal('4.4') * weight_kg * Decimal('3.5')) / Decimal('100')

        YogaAsanaRecord1 = YogaAsanaRecord.objects.filter(fitness_profile=FitnessProfile1,date=today)
        context = {
            'ucount': ucount,
            'aasna': AasnaVideo1,
            'reg': RegisteredUser1,
            'fit': FitnessProfile1,
            'curls': CurlRecord1,
            'squats': SquatRecord1,
            'pushup': PushupRecord1,
            'curlCal': curlCal,
            'squatCal': squatCal,
            'pushupCal': pushupCal,
            'rem': rem,
            'curlbr': curlbr,
            'squatbr': squatbr,
            'pushupbr': pushupbr,
            'yog':YogaAsanaRecord1,
        }
    return render(request, 'dashboard.html', context)


