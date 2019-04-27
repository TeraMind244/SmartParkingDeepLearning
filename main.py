# main.py

from flask import Flask, render_template, Response, abort
from camera import VideoCamera
import cv2
import opencv_identifier as opencv
import image_utils as iu
import read_data as rd
import numpy as np
#import requests

# %%


app = Flask(__name__)

list_slots = []
message = ""

data = rd.get_data()
current_lotId = data['lotId']
cam_list = data['cam_list']

for cam in cam_list:
    cam['transform_matrix'] = []
    cam['list_slots'] = []

# %%


def add_text(image, transform_matrix, list_slots):
    if len(transform_matrix) == 0:
        return image
    else:
        return iu.reverse_image(image, transform_matrix, list_slots)
    
    
#TODO merge result of list slots from camera
    
def request_update(frame, lotId, camId):
    global message
    try:
        transform_matrix, list_slots = opencv.detect_parking_image(frame, camId)
        cam = next(filter(lambda cam: cam['cam_id'] == camId, cam_list))
        cam['transform_matrix'] = transform_matrix
        cam['list_slots'] = list_slots
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


def gen(camera, lotId):
    count = 200
    while True:
        result = []
        for cam in cam_list:
            frame = camera.get_frame(cam['cam_id'])
            
            if count == 200:
#                request_update(frame, lotId)
                count = 0
            
            height, width, depth = frame.shape
            gap = np.full((50, width, depth), 255)
#            frame = add_text(frame, cam['transform_matrix'], cam['list_slots'])
            if len(result) == 0:
                result = frame
            else:
                result = np.concatenate((result, gap, frame), axis=0)

        ret, jpeg = cv2.imencode('.jpg', result)
        count += 1
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# %%

@app.route('/')
def index():
    return render_template('index.html', 
                           lotId=current_lotId)

# %%

@app.route('/video_feed/<int:lotId>', methods=['GET'])
def video_feed(lotId):
    if lotId != current_lotId:
        return abort(404)
    return Response(gen(VideoCamera(cam_list), lotId),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#    return render_template("cam_partial.html", 
#                           cam_list=list(map(lambda cam: cam['cam_id'], data['cam_list'])), 
#                           lotId=current_lotId)
    
#@app.route('/cam_feed/<int:lotId>/<int:camId>', methods=['GET'])
#def cam_feed(lotId, camId):
#    return Response(gen(VideoCamera(camId), lotId, camId),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# %%

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, threaded=False, debug=False, use_reloader=False)
    
    