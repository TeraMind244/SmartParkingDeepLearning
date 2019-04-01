# main.py

from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2
import opencv_identifier as opencv

app = Flask(__name__)

cnt_empty = 0
all_spots = 0

async def request_update(frame):
    global all_spots, cnt_empty
    image, all_spots, cnt_empty = opencv.identify_parking_spot(frame)
    print("Running")
#    try:
#        image, all_spots, cnt_empty = opencv.test_return_image(frame)
#    except:
#        print("Something went wrong!")

def add_text(image, cnt_empty, all_spots):
    cv2.putText(image, "Available: %d spots" %cnt_empty, (30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2)

    cv2.putText(image, "Total: %d spots" %all_spots, (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    return image

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    count = 0
    while True:
        frame = camera.get_frame()
        global cnt_empty, all_spots
#        buffer_image = frame
        if count % 10 == 0:
            request_update(frame)
        frame = add_text(frame, cnt_empty, all_spots)
        ret, jpeg = cv2.imencode('.jpg', frame)
        image = jpeg.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')
        count += 1

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=8080, threaded=False, debug=False, use_reloader=False)
    
    
    
    
    