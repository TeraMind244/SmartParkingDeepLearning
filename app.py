#!flask/bin/python
from flask import Flask, jsonify, abort, make_response, request
import opencv_identifier

app = Flask(__name__)

@app.route('/api/lot', methods=['GET'])
def get_lot():
    return jsonify({'message': "Hello World"})

if __name__ == '__main__':
    app.run(debug=True)