# use https://developers.google.com/mediapipe/solutions/vision/interactive_segmenter
# but have the point of interest set to the mouse cursor while in the window

import mediapipe as mp #live video ML tools
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import InteractiveSegmenter, InteractiveSegmenterOptions, InteractiveSegmenterRegionOfInterest
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
import cv2 #image manipulation
import numpy as np
import copy
import time
import math
ROI = InteractiveSegmenterRegionOfInterest

model_path = 'magic_touch.tflite'
camera_index = 0
camera_warmup = 0.1 #scale to how long it takes for your in use camera
window_name = 'Frame'
model_width = 640
model_height = 480
overlay_colour = (44, 178, 179)
#BG_COLOR = (192, 192, 192) # gray
#MASK_COLOR = (255, 255, 255) # white

def handle_click(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        #print(mouse_x, mouse_y)

mouse_x, mouse_y = model_width // 2, model_height // 2

def get_image(capture, width=model_width, height=model_height): 
    ret, frame = capture.read()
    image = cv2.resize(frame, (width, height))
    return image

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

options = InteractiveSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_category_mask=True
)

video_capture = cv2.VideoCapture(camera_index) #grab feed from first camera
first_frame = get_image(video_capture) #initialise camera, ignore first frame
time.sleep(camera_warmup) #allow camera warmup time

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, handle_click)
with InteractiveSegmenter.create_from_options(options) as segmenter:
    while True:
        image = get_image(video_capture)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #switch blue-green-red image to red-green-blue

        fps = video_capture.get(cv2.CAP_PROP_FPS)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        roi = ROI(format=ROI.Format.KEYPOINT, keypoint=NormalizedKeypoint(mouse_x, mouse_y))
        result = segmenter.segment(mp_image, roi)

        category_mask = result.category_mask

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #return RGB image to BGR for presentation

        #TODO: smooth and use transparent colour
        image_data = image
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = np.flip(overlay_colour)
        
        alpha = 0.5
        fg_image = cv2.addWeighted(image, 1 - alpha, fg_image, alpha, 0)
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        image_out = np.where(condition, image_data, fg_image)

        # Draw a white dot with black border to denote the point of interest
        thickness, radius = 6, -1
        cv2.circle(image_out, (mouse_x, mouse_y), thickness + 5, (0, 0, 0), radius)
        cv2.circle(image_out, (mouse_x, mouse_y), thickness, (255, 255, 255), radius)

        cv2.imshow(window_name, image_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
video_capture.release()