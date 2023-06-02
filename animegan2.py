import torch
import cv2
import time
from torchvision import transforms
from PIL import Image
import numpy as np

model_name = "celeba_distill"
#model_name = "face_paint_512_v1"
#model_name = "face_paint_512_v2"
#model_name = "paprika"

model_width = 640
model_height = 480
camera_index = 0
camera_warmup = 0.1
camera_flip_view = True

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Running on CUDA: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
  print("Running on CPU")
  device = torch.device("cpu")

model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=model_name)
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)

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

    result = face2paint(model, image, side_by_side=True)

    #cv2.imshow("Input", frame)

    result_cv = np.array(result.convert('RGB'), dtype=np.uint8)
    #result_cv = result_cv[:, :, ::-1].copy() #RGB to BGR

    cv2.imshow("AnimeGANv2 " + model_name, result_cv)

    print(time.time() - iter_start)
    # if the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
video_capture.release()