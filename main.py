# main.py

from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2
import opencv_identifier as opencv
import time
import numpy as np

app = Flask(__name__)

list_slots = []
message = ""

def request_update(frame):
    global list_slots, message
    try:
        list_slots = opencv.identify_parking_spot(frame)
    except:
        message = "Something went wrong!"

def add_text(image):

    new_image = np.copy(image)
    overlay = np.copy(image)
    color=[0, 255, 0]
    alpha=0.5
    all_spots = 0
    cnt_empty = 0
    
    height, width, depth = image.shape
    
    scale = 720/height
    
    global list_slots
    
    for slot in list_slots:
        all_spots += 1
        (x1, y1, x2, y2) = slot['position']
        (x1, y1, x2, y2) = (int(x1)/scale, int(y1)/scale,
                             int(x2)/scale, int(y2)/scale)
        
        label = slot['status']

        if label == 'empty':
            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)),
                          color, -1)
            cnt_empty += 1

    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

    cv2.putText(new_image, "Available: %d spots" %cnt_empty, (30, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (255, 255, 255), 2)

    cv2.putText(new_image, "Total: %d spots" %all_spots, (30, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (255, 255, 255), 2)
    
    return new_image

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    count = 0
    while True:
        frame = camera.get_frame()
        if count == 100:
            request_update(frame)
            count = 0
        time.sleep(0.1)
        frame = add_text(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        image = jpeg.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')
        count += 1

@app.route('/video_feed/<int:lotId>', methods=['GET'])
def video_feed(lotId):
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=8081, threaded=False, debug=False, use_reloader=False)
    
    
    
    
    