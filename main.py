# main.py

from flask import Flask, render_template, Response, abort
from camera import VideoCamera
import cv2
import opencv_identifier as opencv
import image_utils as iu
import read_data as rd
import numpy as np
import requests

# %%


app = Flask(__name__)

merged_list_slots = []
message = ""

data = rd.get_data()
current_lotId = data['lotId']
cam_list = data['cam_list']
api_domain = data['api_domain']

for cam in cam_list:
    cam['transform_matrix'] = []
    cam['list_slots'] = []
    cam['scale'] = 1.0

# %%


def add_text(image, transform_matrix, list_slots, scale):
    if len(transform_matrix) == 0:
        return image
    else:
        return iu.reverse_image(image, transform_matrix, list_slots, scale)
    

# %%
        
    
def merge_slots(list_slots):
    global merged_list_slots
    for slot in list_slots:
        row = slot['row']
        lane = slot['lane']
        status = slot['status']
        existed_slot = next(filter(lambda spot: spot['row'] == row and spot['lane'] == lane,
                                   merged_list_slots), {})
        if existed_slot:
            existed_slot['status'] = status
        else:   
            merged_list_slots.append(slot)
    
    
# %%
            
            
def request_update(frame, lotId, camId):
    global message
    try:
#        print('Updating camera' + str(camId))
        transform_matrix, list_slots, scale = opencv.detect_parking_image(frame, camId)
        merge_slots(list_slots)
        cam = next(filter(lambda cam: cam['cam_id'] == camId, cam_list), {})
        if cam:
            cam['transform_matrix'] = transform_matrix
            cam['list_slots'] = list_slots
            cam['scale'] = scale
        
        global merged_list_slots
        if len(merged_list_slots) > 0:
            list_slots_param = list(map(lambda slot: {
                    'row':slot['row'],
                    'lane':slot['lane'],
                    'status':slot['status']
                    }, merged_list_slots
            ))
            requests.put(api_domain + 'public/update_status_slot?parkingLotId=' + str(lotId), 
                         json=list_slots_param)
            
    except:
        message = "Something went wrong!"


# %%
        

def gen(camera, lotId):
    count = 100
    while True:
        result = []
        for cam in cam_list:
            camId = cam['cam_id']
            frame = camera.get_frame(camId)
            
            if count == 100:
                request_update(frame, lotId, camId)
            
            height, width, depth = frame.shape
            gap = np.full((20, width, depth), 255)
            
            frame = add_text(frame, 
                             cam['transform_matrix'], 
                             cam['list_slots'], 
                             cam['scale'])
            
            if len(result) == 0:
                result = frame
            else:
                result = np.concatenate((result, gap, frame), axis=0)

        ret, jpeg = cv2.imencode('.jpg', result)
        if count == 100:
            count = 0
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


# %%

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, 
            threaded=False, debug=False, 
            use_reloader=False)
    
    