from pathlib import Path
import cv2
import json
import time
import picamera
import picamera.array

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

import numpy as np
from backends.yn_backend import YNBackend

DIRECTORY="./CAMERA_TESTS/"
SCORE_THRESHOLD=0.5
#RESOLUTION=(2592,1944)
RESOLUTION=(2592,1952)
RESOLUTION=(4056,3040) #4056x3040
TIMER_LIMIT=300


# Function to capture an image from the camera and return it as a NumPy array
def capture_image_as_array():

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = RESOLUTION

    rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    time.sleep(0.1)
    # grab an image from the camera
    camera.capture(rawCapture, format="rgb")
    image_array = rawCapture.array

    camera.close()
# camera.awb_mode = 'auto'  # Adjust white balance if needed
    return image_array
    
# ML inference function
def ml_inference(inp_file,inp_name):


    d = YNBackend(score_threshold=SCORE_THRESHOLD)
    #img = cv2.imencode('.jpg', inpfile)[1].tobytes()
    print("IMG SHAPE", inp_file.shape)

    # save the captured image
    image_name = inp_name
    
    #cv2.imwrite(image_name, inpfile)
    
    # load the captured image using cv2.imread()
    #loaded_image = cv2.imread(inpfile)
    
    # img = cv2.imread(inpfile)

    start = time.time()
    res = d.run(inp_file)
    end = time.time()

    DURATION=end - start
    
    if(res is not None):

        cv2.imwrite(DIRECTORY+image_name, inp_file)
        
        outfile = Path(DIRECTORY+image_name[:-4]).with_suffix('.det')
        with open(outfile, 'w') as f:
            json.dump(res, f)
            print(f'Wrote JSON to {outfile}')
                
        DETECTION_COUNT=len(res)
    else:
        DETECTION_COUNT=0
        
    print("[INFO] face detection found {} faces and took {:.4f} seconds".format(DETECTION_COUNT, DURATION))
    
    
    # Perform your ML inference here
    # Replace this with your actual implementation
    return res


def capture_image():
    last_capture_time = 0
        
    while True:
        current_time = time.time()
        
        # Capture image if 5 seconds have elapsed since the last capture
        if current_time - last_capture_time >= TIMER_LIMIT:
            # Capture image from camera
            captured_image = capture_image_as_array()
            image_name = f"FN07_{time.strftime('%Y-%m-%d_%H-%-M-%S')}.jpg"
            print(f"Captured image: {image_name}")

            # Update the last capture time
            last_capture_time = current_time
            
            # Call ML inference function
            result = ml_inference(captured_image, image_name)
            
            if result is not None and result != "":
                # Inference returned a non-blank value
                #break
                continue
        
        # Wait for 1 second before checking again
        time.sleep(1)
                
     
           
if __name__ == '__main__':
    capture_image()
