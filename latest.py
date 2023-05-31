"""
point prediction code adapted for Sussex open day by Dexter 
"""
import mediapipe as mp #provide ML tools
import time 
import cv2 #provide image manipulation
import copy as cp

"""
gets the image and size it to model parameters
@param: cam camera input
@return: im camera image in an array
"""
def getImg(cam): 
    ret, frame = cam.read()
    im=cv2.resize(frame,(WIDTH, HEIGHT))
    return im

"""
gets a person based on landmarks
@param: im shot of the camera in an image form
@return: image a cropped image
@retun: results = a processed image
"""
def get_person(im):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    im.flags.writeable = False
    results = pose.process(im)
    # Draw the pose annotation on the image.
    im.flags.writeable = True
    return im,results

"""
crops a person out of the image
@param: image 
@param: results
@return: image
"""
def crop_person(image,results):
    # clear the stream in preparation for the next frame
        try:
            landmarks=results.pose_landmarks.landmark
            c=0
            points=[7,11,23,25,31] #points of interest
            while landmarks[c].visibility>0.9 and c<len(landmarks): #count how 
                c+=1
            EndC=7
            for i in points: #loop through minimal point
                if c<i:
                    EndC=i
            #get forst and last points
            x=int(WIDTH*landmarks[8].x)
            y=int(HEIGHT*landmarks[8].y)
            ex=int(WIDTH*landmarks[EndC].x)
            ey=int(HEIGHT*landmarks[EndC].y)
            image[max(y-130,0):min(HEIGHT,ey+30),max(x-130,0):min(WIDTH,ex+30)]=(0,0,0)
        except AttributeError:
            pass
        return image
    

mp_drawing = mp.solutions.drawing_utils
mp_pose_user = mp.solutions.pose

# initialize the camera and grab a reference to the raw camera capture
cam = cv2.VideoCapture(0) ####MAKE SURE THE INTEGER IS TO THE RIGHT PORT
#default height and width
HEIGHT=480
WIDTH=640
pose=mp_pose_user.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) #create pose object
    
rawCapture = getImg(cam)
# allow the camera to warmup
time.sleep(0.1)

ppl=[]
similarity=0.0
# capture frames from the camera
while True:
        image = getImg(cam) #get the image
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        im=image.copy() #create copy
        people=True
        MAX=2
        i=0
        res=[]
        while people and i<MAX: #detect multiple people
            im,results=get_person(im)
    
            # draw landmarks on each of the points
            if results.pose_landmarks!=None: #draw next person
                im=crop_person(im,results).copy()
                ppl.append(im)
                res.append(cp.deepcopy(results))
                
            else: people=False
            i+=1
        print(len(res))
        if len(ppl) > 1:
            errorL2 = cv2.norm(ppl[0], ppl[1], cv2.NORM_L2 )
            similarity = 1 - errorL2 / ( HEIGHT * WIDTH )
            print('Similarity = ',similarity)

        if similarity  > 0.75:
            for i in range(len(res)):
                mp_drawing.draw_landmarks(image, res[i].pose_landmarks, mp_pose_user.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                            )
        else:
            if len(res) > 0:
                mp_drawing.draw_landmarks(image, res[0].pose_landmarks, mp_pose_user.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                            )
            else:
                print('nothing detected')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #correct form
        cv2.imshow("Frame cropps", im) #show the image
        cv2.imshow("Frame", image) #show the image
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
cam.release()
     
