import torch
import cv2
import time
from torchvision import transforms
from PIL import Image
import numpy as np

model_width = 640
model_height = 480
camera_index = 0
camera_warmup = 0.1
camera_flip_view = True

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Running on CUDA:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
  print("Running on CPU")
  device = torch.device("cpu")

model_celeba = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="celeba_distill", device=device)
model_fp512_2 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2", device=device)
model_paprika = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika", device=device)

face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)

model_index = 0
models = [model_celeba, model_fp512_2, model_paprika]
model_names = ["celeba_distill", "face_paint_512_v2", "paprika"]

LINE_CLEAR = '\x1b[2K'

#grab next frame from capture device
def get_image(capture, width=model_width, height=model_height): 
    ret, frame = capture.read()
    image = cv2.resize(frame, (width, height))
    if camera_flip_view:
       image = cv2.flip(image, 1)
    return image

video_capture = cv2.VideoCapture(camera_index) #grab feed from first camera
first_frame = get_image(video_capture) #initialise camera, ignore first frame
time.sleep(camera_warmup) #allow camera warmup time
timestamp = 0
while True:
    iter_start = time.time()
    # Load the input image.
    frame = get_image(video_capture)
    image = Image.fromarray(frame)

    model = models[model_index]

    result = face2paint(model, image, side_by_side=True)

    result_cv = np.array(result.convert('RGB'), dtype=np.uint8)
    #result_cv = result_cv[:, :, ::-1].copy() #RGB to BGR

    cv2.imshow("AnimeGANv2: Primitive Style Transfer - Press Q to Quit, A and D to Switch Styles", result_cv)

    print((int) (1 // (time.time() - iter_start)), 'FPS', ', Model: ' + model_names[model_index], end='\r') #FPS counter with carriage return
    # if the 'q' key was pressed, break from the loop
    key_byte = cv2.waitKey(1) & 0xFF
    if key_byte == ord("q"):
      break
    elif key_byte == ord('a'): #left
       model_index = len(models) - 1 if model_index == 0 else model_index - 1
       print(end=LINE_CLEAR) # clear line so that FPS counter/model name isn't ghosted (part left behind if shorter than previous line)
    elif key_byte == ord('d'): #right
       model_index = 0 if model_index == len(models) - 1 else model_index + 1
       print(end=LINE_CLEAR) # clear line so that FPS counter/model name isn't ghosted (part left behind if shorter than previous line)

print(end='\n') # allow user to see final FPS/model on termination
cv2.destroyAllWindows()
video_capture.release()