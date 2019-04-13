# camera.py

import cv2
#import time
#import opencv_identifier as opencv

class VideoCamera(object):
    
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(1)
    
    def __del__(self):
        self.video.release()
                
    def get_frame(self):
        success, frame = self.video.read()
        return frame
    