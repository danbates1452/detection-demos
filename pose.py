import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
#from scipy.signal import convolve2d
import time

#model_path = 'pose_landmarker_lite.task'
#model_path = 'pose_landmarker_heavy.task'
model_path = 'pose_landmarker_full.task'
min_detect_conf = 0.5
min_prescence_conf = 0.5
min_track_conf = 0.5

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

"""
kernel = np.add.outer(*2*(np.arange(3) % 2,))**2 / 8
def perfect_edges(orig, n_iter=1, thresh=20):
    mask = orig <= thresh
    corrector = convolve2d(mask, kernel, 'same')
    result = orig.copy()
    result[mask] = 0
    for j in range(n_iter):
        result = result * corrector + convolve2d(result, kernel, 'same')
        result[mask] = 0
    result = np.round(result).astype(np.uint8)
    result[mask] = orig[mask]
    return result
"""

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
    timestamp += int(fps)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect pose landmarks from the input image.
    detection_result = detector.detect_for_video(image, timestamp)

    # Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    #cv2.imshow('Annotated Frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Annotated Frame', annotated_image)
    if detection_result.segmentation_masks is not None:
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        #segmentation_mask = perfect_edges(segmentation_mask, smoothing_iter, smoothing_thresh)
        #visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        #cv2.imshow('Visualised Mask', visualized_mask)

        #blurred_image = cv2.GaussianBlur(frame, (55,55), 0)
        condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
        #blurred_mask = np.where(condition, blurred_image, frame)
        #cv2.imshow('Blurred Mask', blurred_mask)

        overlaid_image = overlay_image(image.numpy_view(), segmentation_mask)
        cv2.imshow('Pose Highlight', overlaid_image)

    # if the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
video_capture.release()