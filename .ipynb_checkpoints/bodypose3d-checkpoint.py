import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, write_keypoints_to_disk
#add here if you need more keypoints


#this will load the sample videos if no camera ID is given
Nvideo = '12'
Calib_n = '11'

Video_nameA = '.\Videos\pcte'+Nvideo+'A.avi'
Video_nameB = '.\Videos\pcte'+Nvideo+'B.avi'
input_stream1 = Video_nameA
input_stream2 = Video_nameB
print(input_stream1)

pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28, 32, 31]
Npoints = len(pose_keypoints)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


Calib_video = '.\Calibration_Videos_RS\Calibration_parameters_'+Calib_n+'.npz'
npz = np.load(Calib_video)


P0=npz['P1']
P1=npz['P2']
R=npz['R']
T=npz['T']


#create body keypoints detector objects.
#INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
pose0 = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.9,model_complexity=2)
pose1 = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.9,model_complexity=2)

#containers for detected keypoints for each camera. These are filled at each frame.
#This will run you into memory issue if you run the program without stop
kpts_cam0 = []
kpts_cam1 = []
kpts_3d = []

## Start reading the videos
cap0 = cv.VideoCapture(input_stream1)
cap1 = cv.VideoCapture(input_stream2)
caps = [cap0, cap1]


while(cap0.isOpened()):
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1: 
        break
    else:
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)
    
        #reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
    
        #check for keypoints detection
        frame0_keypoints = []
        
        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                if i not in pose_keypoints: continue #only save keypoints that are indicated in pose_keypoints
                    
                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame0,(pxl_x, pxl_y), 3, (0,0,255), -1) #add keypoint detection points into figure
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts) 
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*Npoints
                
        #this will keep the keypoints of this frame in memory
        kpts_cam0.append(frame0_keypoints)

        ## For the second camera
        frame1_keypoints = []
        if results1.pose_landmarks:
            for i, landmark in enumerate(results1.pose_landmarks.landmark):
                    
                if i not in pose_keypoints: continue
                        
                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame1,(pxl_x, pxl_y), 3, (0,0,255), -1)
                kpts = [pxl_x, pxl_y]
                frame1_keypoints.append(kpts)
    
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]]*len(pose_keypoints)
    
        #update keypoints container
        kpts_cam1.append(frame1_keypoints)
        
        
        ##calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)
                
        frame_p3ds = np.array(frame_p3ds).reshape((Npoints, 3))
        kpts_3d.append(frame_p3ds)
        # uncomment these if you want to see the full keypoints detections
        mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Show reults
        frame_cat = cv.hconcat([frame0,frame1])
        half = cv.resize(frame_cat, (0, 0), fx = 0.5, fy = 0.5)
        cv.imshow('cam0', half)
      
        if cv.waitKey(1) & 0xFF == ord('q'):
            break #27 is ESC key.
    
for cap in caps:
    cap.release()
cv.destroyAllWindows()

#this will create keypoints file in current working folder
write_keypoints_to_disk('.\Tracking\kpts_cam0_'+Nvideo+'.dat', kpts_cam0)
write_keypoints_to_disk('.\Tracking\kpts_cam1_'+Nvideo+'.dat', kpts_cam1)
write_keypoints_to_disk('.\Tracking\kpts_3d_'+Nvideo+'.dat', kpts_3d)
print('All done')