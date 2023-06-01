import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import copy

#model_path = 'pose_landmarker_lite.task'
#model_path = 'pose_landmarker_heavy.task'
model_path = 'pose_landmarker_full.task'
min_detect_conf = 0.5
min_prescence_conf = 0.5
min_track_conf = 0.5
max_people = 2

model_width = 640
model_height = 480
camera_index = 0
camera_warmup = 0.1
camera_flip_view = True
smoothing_iter = 1
smoothing_thresh = 14
overlay_colour = (44, 178, 179)

#overlay a transparent coloured layer on an image based on a 2d mask
def overlay_image(image_data, overlay_mask, colour=overlay_colour, alpha_val=0.5):
    overlay_image = np.zeros(image_data.shape, dtype=np.uint8)
    overlay_image[:] = np.flip(colour)
        
    alpha = np.stack((overlay_mask,) * 3, axis=-1) > 0.1
    alpha = alpha.astype(float) * alpha_val

    image_out = image_data * (1 - alpha) + overlay_image * alpha
    return image_out.astype(np.uint8)

#grab next frame from capture device
def get_image(capture, width=model_width, height=model_height): 
    ret, frame = capture.read()
    image = cv2.resize(frame, (width, height))
    if camera_flip_view:
       image = cv2.flip(image, 1)
    return image

#source: https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image   

# Crop a person from a frame so that other people can be detected as well
def crop_person(image, results, width=model_width, height=model_height):
    try:
        landmarks=results.pose_landmarks
        """
        7 - left ear
        11 - left shoulder
        23 - left hip
        25 - left knee
        31 - left foot index
        """
        points=[7,11,23,25,31] #points of interest
        landmark_counter=0
        while landmarks[landmark_counter].visibility > 0.9 and c < len(landmarks):
            c+=1 #count up significant landmarks (above 0.9 visibility)
        EndC=7
        for i in points: #loop through minimal point
            if c<i:
                EndC=i
        #get first and last points
        x=int(width*landmarks[8].x)
        y=int(height*landmarks[8].y)
        ex=int(width*landmarks[EndC].x)
        ey=int(height*landmarks[EndC].y)
        image[max(y-130,0):min(height,ey+30),max(x-130,0):min(width,ex+30)]=(0,0,0)
    except AttributeError:
        print("no worky")
        pass
    return image

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=True,
    min_pose_detection_confidence=min_detect_conf, 
    min_pose_presence_confidence=min_prescence_conf,
    min_tracking_confidence=min_track_conf
    )
    
detector = vision.PoseLandmarker.create_from_options(options)

video_capture = cv2.VideoCapture(camera_index) #grab feed from first camera
first_frame = get_image(video_capture) #initialise camera, ignore first frame
time.sleep(camera_warmup) #allow camera warmup time
timestamp = 0
fps = video_capture.get(cv2.CAP_PROP_FPS)
while True:
    # Load the input image.
    frame = get_image(video_capture)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    crop_image = copy.copy(frame)
    annotated_image = copy.copy(frame)
    overlaid_image = copy.copy(frame)

    people_left = True
    people_detected = 0
    while people_left and people_detected < max_people:
        # Detect pose landmarks from the input image.
        detection_result = detector.detect_for_video(image, timestamp)    
        
        if detection_result.pose_landmarks is not None:
            # crop person
            crop_image = crop_person(crop_image, detection_result)

            # add annotation
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

            # if segmentation mask returned, add overlay
            if detection_result.segmentation_masks is not None:
                segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
                overlaid_image = overlay_image(image.numpy_view(), segmentation_mask)
        else:
            people_left = False
        people_detected += 1
        timestamp += int(fps)
    cv2.imshow('Annotated Frame', annotated_image)
    cv2.imshow('Pose Highlight', overlaid_image)
    cv2.imshow('Crops', crop_image) #debugging only

    # if the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
video_capture.release()