# camera.py

import cv2
import numpy as np
#import image_utils as iu
#import time
#import opencv_identifier as opencv

class VideoCamera(object):
    
    cam_list = []
    
    def __init__(self, cam_list):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment these lines below out and use a video file
        # instead.
#        self['video' + str(camId)] = cv2.VideoCapture(camId)
#        for cam in self.cam_list:
#            cam['video'] = cv2.VideoCapture(cam['cam_id'])
#        self.cam_list = list(map(lambda cam: 
#                                    cam['video'] = cv2.VideoCapture(cam['cam_id'])
#                                , cam_list))
        for cam in cam_list:
            cam['video'] = cv2.VideoCapture(cam['cam_id'])
            self.cam_list.append(cam)
    
    def __del__(self):
        for cam in self.cam_list:
            cam['video'].release()
#        self.video.release()
                
    def get_frame(self, camId):
        camera = (next(filter(lambda cam: cam['cam_id'] == camId, self.cam_list)))['video']
        success, frame = camera.read()
        if success:
            return frame
        else:
#            return white image
            return np.zeros_like((480, 640, 3))
        