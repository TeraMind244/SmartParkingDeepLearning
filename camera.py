# camera.py

import cv2
#import time
#import opencv_identifier as opencv

class VideoCamera(object):
    
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture('outpy.avi')
#        self.buffer_frame_in = []
#        self.buffer_frame_out = []
#        self.cnt_empty = 2
#        self.all_spots = 20
#        self.running = False
#        self.count = 1
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
#        self.running = False
        
#    def add_text(self, image):
#        cv2.putText(image, "Available: %d spots" %self.cnt_empty, (30, 35),
#                cv2.FONT_HERSHEY_SIMPLEX,
#                0.7, (255, 255, 255), 2)
#
#        cv2.putText(image, "Total: %d spots" %self.all_spots, (30, 65),
#                    cv2.FONT_HERSHEY_SIMPLEX,
#                    0.7, (255, 255, 255), 2)
#        return image
        
#    async def set_interval(self):
#        while True:
#            time.sleep(1)
##            self.buffer_frame_out = image
#            self.cnt_empty = self.count + 1
#            self.all_spots = self.count * 4 + 1
#            self.count += 1
#            self.buffer_frame, self.all_spots, self.cnt_empty = await opencv.test_return_image(image)
                
    def get_frame(self):
        success, frame = self.video.read()
#        self.buffer_frame_in = image
#        frame = opencv.identify_parking_spot(None)
#        if not self.running:
##            self.buffer_frame_out = image
#            self.set_interval()
#            self.running = True
        
#        frame = self.add_text(frame)
        
        
        return frame
    