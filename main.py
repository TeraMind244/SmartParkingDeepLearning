# main.py

from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2
import opencv_identifier as opencv
import image_utils as iu
import read_data as rd
#import numpy as np
#import requests

app = Flask(__name__)

list_slots = []
transform_matrix = []
message = ""

data = rd.get_data()

current_lotId = data['lotId']

def request_update(frame, lotId):
    global list_slots, message, transform_matrix
    try:
        transform_matrix, list_slots = opencv.detect_parking_image(frame)
#        print(list_slots)
#        if len(list_slots) > 0:
#            list_slots_param = list(map(lambda slot: {
#                    'row':slot['row'],
#                    'lane':slot['lane'],
#                    'status':slot['status']
#                    }, list_slots
#            ))
#            requests.put('http://localhost:8080/public/update_status_slot?parkingLotId=' + str(lotId), 
#                         json=list_slots_param)
    except:
        message = "Something went wrong!"

def add_text(image):
    if len(transform_matrix) == 0:
        return image
    else:
        return iu.reverse_image(image, transform_matrix, list_slots)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera, lotId):
    count = 100
    while True:
        frame = camera.get_frame()
        if count == 100:
            request_update(frame, lotId)
            count = 0
#        time.sleep(0.1)
        frame = add_text(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        image = jpeg.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')
        count += 1

@app.route('/video_feed/<int:lotId>', methods=['GET'])
def video_feed(lotId):
    if lotId != current_lotId:
        return Response()
    return Response(gen(VideoCamera(), lotId),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, threaded=False, debug=False, use_reloader=False)
    
    