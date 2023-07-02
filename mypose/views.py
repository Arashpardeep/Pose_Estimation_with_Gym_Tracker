import sys
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create your views here.
def signin(request):
    if not request.user.is_authenticated:

        if request.method == 'POST':
            username = request.POST['username']
            pass1 = request.POST['pass1']
            user = authenticate(username = username, password = pass1)

            if user is not None:
                login(request, user)
                fname = user.first_name
                messages.success(request, f"You are logged in successfully as {username}")
                request.session['username'] = True
                return render(request, 'home.html', {'username':username})
            else:
                messages.error(request, "User is not registered")
                return redirect('signin')

        return render(request, 'login.html')

    else:
        messages.error(request, "You are already logged in")
        return redirect('index')

def Signup(request):
    if request.method == "POST":    
        username = request.POST['username'] 
        fname = request.POST['fname'] 
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        if User.objects.filter(username = username):
            messages.error(request, "Username already exist! Please try some other username")
            return redirect('signup')

        if User.objects.filter(email = email):
            messages.error(request, "Email already registered! Please login to your account!")
            return render(request, 'login.html')

        if len(username)>15:
            messages.error(request, "Username must be under 15 characters")
            return redirect('signup')

        if pass1 != pass2:
            messages.error(request, "Passwords didn't match")
            return redirect('signup')

        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!")
            return redirect('signup')

        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.save()
        messages.success(request, "Your account has been successfully created!")

        return redirect('signin') 

    return render(request, 'Signup.html')

def signout(request):
    logout(request)
    messages.success(request, "Logged Out successfully!")
    return redirect('signin')

def index(request):
    myuser = request.user
    username = myuser.username
    if request.user.is_authenticated:
        return render(request, 'home.html', {'username':username})
    else:
        messages.error(request, "Anonymous users are not allowed to enter into the website! Please do login first")
        return redirect('signin')
    
def blog(request):
    myuser = request.user
    username = myuser.username
    return render(request, 'blog.html', {'username':username})

def landmarks(request):
    myuser = request.user
    username = myuser.username
    return render(request, 'Landmarks.html', {'username':username})
    
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def webcam(request):
    myuser = request.user
    username = myuser.username
    return render(request, 'exercises.html', {'username':username})
    
def bicep(request):
    myuser = request.user
    username = myuser.username

    cap = cv2.VideoCapture(0)

    # Curl Counter variables
    counter1 = 0
    stage1 = None
    counter2 = 0
    stage2 = None

    # Setup Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # RECOLOR IMAGE TO RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # We do this because when we pass an image to mediapipe, we want the image to be in format of RGB, but when we get feed using OpenCV, by default, the image feed is going to be in the format of BGR

            # MAKE DETECTION
            results = pose.process(image)

            # RECOLOR BACK TO BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get Coordinates
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                # Calculate Angle
                angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

                # Visualize angle
                cv2.putText(image, str(angle_left),
                            tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                cv2.putText(image, str(angle_right),
                            tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                
                # Curl Counter Logic
                if angle_left > 160:
                    stage1 = "down"
                if angle_left < 50 and stage1 == 'down':
                    stage1 = "up"
                    counter1 += 1
                    
                if angle_right > 160:
                    stage2 = "down"
                if angle_right < 50 and stage2 == 'down':
                    stage2 = "up"
                    counter2 += 1
                    
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (255, 115), (245, 117, 16), -1)
            cv2.rectangle(image, (1000, 115), (0, 0), (245, 117, 16), -1)


            # Rep data for left angle
            cv2.putText(image, 'Exercise: Bicep Curls', (15, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA
                    )
            cv2.putText(image, 'LEFT ARM', (15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 225, 255), 1, cv2.LINE_AA
                        )
            cv2.putText(image, 'REPS', (15, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter1), (10, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            # Stage data for left angle
            cv2.putText(image, 'STAGE', (85, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, stage1, (80, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Rep data for right angle
            cv2.putText(image, 'RIGHT ARM', (450, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 225, 255), 1, cv2.LINE_AA
                        )
            cv2.putText(image, 'REPS', (450, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter2), (440, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            # Stage data for right angle
            cv2.putText(image, 'STAGE', (520, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, stage2, (510, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2, cv2.LINE_AA
                        )


            # RENDER DETECTIONS
            mp_drawing.draw_landmarks(image, 
                                    results.pose_landmarks, 
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2,circle_radius = 2),
                                    mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2,circle_radius = 2),
                                    )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'exercises.html', {'username':username})

def squat(request):
    myuser = request.user
    username = myuser.username

    cap = cv2.VideoCapture(0)

    # Curl Counter variables
    counter1 = 0
    counter2 = 0
    counter3 = 0 
    stage1 = None

    flag = 0

    # Setup Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # RECOLOR IMAGE TO RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # We do this because when we pass an image to mediapipe, we want the image to be in format of RGB, but when we get feed using OpenCV, by default, the image feed is going to be in the format of BGR

            # MAKE DETECTION
            results = pose.process(image)

            # RECOLOR BACK TO BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get Coordinates
                hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


                # Calculate Angle
                angle_left = calculate_angle(hip_left, knee_left, ankle_left)
                angle_right = calculate_angle(hip_right, knee_right, ankle_right)
                avg_angle = (angle_left + angle_right) / 2

                # Visualize angle
                cv2.putText(image, str(angle_left),
                            tuple(np.multiply(knee_left, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                cv2.putText(image, str(angle_right),
                            tuple(np.multiply(knee_right, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                
                # Curl Counter Logic
                if counter1 == 0 and avg_angle > 170:
                    stage1 = "Not started yet"
                if avg_angle > 170 and stage1 != 'Not started yet' and flag == 0:
                    stage1 = "finished"
                    counter3 += 1
                    flag = 1
                if avg_angle > 170 and stage1 == 'started' and flag == 1:
                    stage1 = "Incomplete Squat"
                    counter2 += 1
                if avg_angle < 165 and (stage1 == 'finished' or stage1 == 'Not started yet' or stage1 == 'Incomplete Squat'):
                    stage1 = "started"
                    counter1 += 1
                    flag = 1
                if stage1 == 'started' and avg_angle < 70:
                    flag = 0

                    
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (255, 110), (245, 117, 16), -1)
            cv2.rectangle(image, (1000, 110), (0, 0), (245, 117, 16), -1)


            # Rep data for left angle
            cv2.putText(image, 'Exercise: Squats', (15, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Total Squats', (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter1), (10, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Incomplete Squats', (130, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter2), (125, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Complete Squats', (300, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter3), (295, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            
            # RENDER DETECTIONS
            mp_drawing.draw_landmarks(image, 
                                    results.pose_landmarks, 
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2,circle_radius = 2),
                                    mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2,circle_radius = 2),
                                    )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'exercises.html', {'username':username})

def pushup(request):
    myuser = request.user
    username = myuser.username

    cap = cv2.VideoCapture(0)

    # Curl Counter variables
    counter1 = 0
    counter2 = 0
    counter3 = 0 
    stage1 = None

    flag = 0

    # Setup Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # RECOLOR IMAGE TO RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # We do this because when we pass an image to mediapipe, we want the image to be in format of RGB, but when we get feed using OpenCV, by default, the image feed is going to be in the format of BGR

            # MAKE DETECTION
            results = pose.process(image)

            # RECOLOR BACK TO BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get Coordinates
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                # Calculate Angle
                angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                avg_angle = (angle_left + angle_right) / 2

                # Visualize angle
                cv2.putText(image, str(angle_left),
                            tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                cv2.putText(image, str(angle_right),
                            tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                
                # Curl Counter Logic
                if counter1 == 0 and avg_angle > 170:
                    stage1 = "Not started yet"
                if avg_angle > 170 and stage1 != 'Not started yet' and flag == 0:
                    stage1 = "finished"
                    counter3 += 1
                    flag = 1
                if avg_angle > 170 and stage1 == 'started' and flag == 1:
                    stage1 = "Incomplete PushUp"
                    counter2 += 1
                if avg_angle < 165 and (stage1 == 'finished' or stage1 == 'Not started yet' or stage1 == 'Incomplete PushUp'):
                    stage1 = "started"
                    counter1 += 1
                    flag = 1
                if stage1 == 'started' and avg_angle < 140:
                    flag = 0

                    
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (255, 110), (245, 117, 16), -1)
            cv2.rectangle(image, (1000, 110), (0, 0), (245, 117, 16), -1)


            # Rep data for left angle
            cv2.putText(image, 'Exercise: PushUps', (15, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Total PushUps', (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter1), (10, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Incomplete PushUps', (150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter2), (145, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Complete PushUps', (330, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter3), (325, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            
            # RENDER DETECTIONS
            mp_drawing.draw_landmarks(image, 
                                    results.pose_landmarks, 
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2,circle_radius = 2),
                                    mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2,circle_radius = 2),
                                    )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'exercises.html', {'username':username})

def lat(request):
    myuser = request.user
    username = myuser.username

    cap = cv2.VideoCapture(0)

    # Curl Counter variables
    counter1 = 0
    counter2 = 0
    counter3 = 0 
    stage1 = None

    flag = 0

    # Setup Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # RECOLOR IMAGE TO RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # We do this because when we pass an image to mediapipe, we want the image to be in format of RGB, but when we get feed using OpenCV, by default, the image feed is going to be in the format of BGR

            # MAKE DETECTION
            results = pose.process(image)

            # RECOLOR BACK TO BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get Coordinates
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate Angle
                angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                avg_angle = (angle_left + angle_right) / 2

                # Visualize angle
                cv2.putText(image, str(angle_left),
                            tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                cv2.putText(image, str(angle_right),
                            tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                
                # Curl Counter Logic
                if counter1 == 0 and avg_angle > 165:
                    stage1 = "Not started yet"
                if avg_angle > 165 and stage1 != 'Not started yet' and flag == 0:
                    stage1 = "finished"
                    counter3 += 1
                    flag = 1
                if avg_angle > 165 and stage1 == 'started' and flag == 1:
                    stage1 = "Incomplete LatPullDown"
                    counter2 += 1
                if avg_angle < 155 and (stage1 == 'finished' or stage1 == 'Not started yet' or stage1 == 'Incomplete LatPullDown'):
                    stage1 = "started"
                    counter1 += 1
                    flag = 1
                if stage1 == 'started' and avg_angle < 70:
                    flag = 0

                    
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (255, 110), (245, 117, 16), -1)
            cv2.rectangle(image, (1000, 110), (0, 0), (245, 117, 16), -1)


            # Rep data for left angle
            cv2.putText(image, 'Exercise: Lat PullDown', (15, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Total LatPullDowns', (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter1), (10, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Incomplete LatPullDowns', (180, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter2), (175, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Complete LatPullDowns', (400, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter3), (395, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            
            # RENDER DETECTIONS
            mp_drawing.draw_landmarks(image, 
                                    results.pose_landmarks, 
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2,circle_radius = 2),
                                    mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2,circle_radius = 2),
                                    )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'exercises.html', {'username':username})

def shoulder(request):
    myuser = request.user
    username = myuser.username

    cap = cv2.VideoCapture(0)

    # Curl Counter variables
    counter1 = 0
    counter2 = 0
    counter3 = 0 
    stage1 = None

    flag = 0

    # Setup Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # RECOLOR IMAGE TO RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # We do this because when we pass an image to mediapipe, we want the image to be in format of RGB, but when we get feed using OpenCV, by default, the image feed is going to be in the format of BGR

            # MAKE DETECTION
            results = pose.process(image)

            # RECOLOR BACK TO BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get Coordinates
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate Angle
                angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                avg_angle = (angle_left + angle_right) / 2

                # Visualize angle
                cv2.putText(image, str(angle_left),
                            tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                cv2.putText(image, str(angle_right),
                            tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                
                # Curl Counter Logic
                if counter1 == 0 and avg_angle > 165:
                    stage1 = "Not started yet"
                if avg_angle < 50 and stage1 == 'started' and flag == 0:
                    stage1 = "finished"
                    counter3 += 1
                    flag = 1
                if avg_angle < 50 and stage1 == 'started' and flag == 1:
                    stage1 = "Incomplete ShoulderPress"
                    counter2 += 1
                if avg_angle > 65 and (stage1 == 'finished' or stage1 == 'Not started yet' or stage1 == 'Incomplete ShoulderPress'):
                    stage1 = "started"
                    counter1 += 1
                    flag = 1
                if stage1 == 'started' and avg_angle > 165:
                    flag = 0

                    
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (255, 110), (245, 117, 16), -1)
            cv2.rectangle(image, (1000, 110), (0, 0), (245, 117, 16), -1)


            # Rep data for left angle
            cv2.putText(image, 'Exercise: Shoulder Press', (15, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Total ShoulderPresses', (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter1), (10, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Incomplete ShoulderPresses', (220, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter2), (210, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, 'Complete ShoulderPresses', (440, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )
            cv2.putText(image, str(counter3), (435, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            
            # RENDER DETECTIONS
            mp_drawing.draw_landmarks(image, 
                                    results.pose_landmarks, 
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2,circle_radius = 2),
                                    mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2,circle_radius = 2),
                                    )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return render(request, 'exercises.html', {'username':username})


def profile(request):
    myuser = request.user
    username = myuser.username
    fname = myuser.first_name
    lname = myuser.last_name
    email = myuser.email
    return render(request, 'profile.html', {'username':username, 'fname':fname, 'lname': lname, 'email': email})
