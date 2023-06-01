from ultralytics import YOLO
import cv2
import time

model_width = 640
model_height = 480
camera_index = 0
camera_warmup = 0.1
camera_flip_view = True

model = YOLO("yolov8n.pt")

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
    # Load the input image.
    frame = get_image(video_capture)
    
    # Run YOLOv8 on it
    results = model(frame)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    # if the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
video_capture.release()