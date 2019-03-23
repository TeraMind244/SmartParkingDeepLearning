#!flask/bin/python
from flask import Flask, jsonify, request, render_template
import opencv_identifier as opencv
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def get_view():
    return render_template('testAPI.html')

@app.route('/lot', methods=['POST'])
def get_lot():
    spots = {'message': 'error'}
    try:
        imageStr = request.files.get('lot-image', '').read()
        npimg = np.fromstring(imageStr, np.uint8)
        img = cv2.imdecode(npimg, -1)
        spots = opencv.identify_parking_spot(img)
    except Exception as err:
        print(err)
    return jsonify(spots)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)