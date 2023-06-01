import mediapipe as mp #live video ML tools
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions, RunningMode
import cv2 #image manipulation
import numpy as np
import copy
import time

model_path = "deeplab_v3.tflite"
camera_index = 0
camera_warmup = 0.1 #scale to how long it takes for your in use camera
model_width = 640
model_height = 480

def get_image(capture, width=model_width, height=model_height): 
    ret, frame = capture.read()
    image = cv2.resize(frame, (width, height))
    return image

options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    output_category_mask=True
)

video_capture = cv2.VideoCapture(camera_index) #grab feed from first camera
first_frame = get_image(video_capture) #initialise camera, ignore first frame
time.sleep(camera_warmup) #allow camera warmup time
timestamp = 0
with ImageSegmenter.create_from_options(options) as segmenter:
    while True:
        frame = get_image(video_capture)
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB) #switch blue-green-red image to red-green-blue

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        timestamp += int(fps)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = segmenter.segment_for_video(mp_image, timestamp)

        category_mask = result.category_mask

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #return RGB image to BGR for presentation

        #TODO: smooth and use transparent colour instead of blur
        blurred_image = cv2.GaussianBlur(frame, (55,55), 0)
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        frame = np.where(condition, frame, blurred_image)

        
        cv2.imshow(str(fps) + " FPS", frame)
        cv2.imshow('Mask', category_mask.numpy_view())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
video_capture.release()