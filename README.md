# Smart Parking

### Contents:

1. `test_images` - Sample image to train the code on

2. `train_data` - Training data for CNN model

3. `camera.py` - Python class for getting frames from your webcam.

Change
```
self.video = cv2.VideoCapture(1)
```
to
```
self.video = cv2.VideoCapture(0)
```
if you don't have external webcam

4. `opencv_identifier.py` - Python script for parking spot detection

5. `main.py` - Python script using Flask to provide API

### Installation

1. You would need Python 2.7 or 3.6 [here](https://www.python.org/downloads/release/python-368/)

2. Please follow [this instruction](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) to install `tensorflow` or `tensorflow-gpu`, depends on your hardware.

3. Install `opencv2`
```
pip install opencv
```
or 
```
conda install opencv
```

4. Install `flask` version `0.12.2`. Please note that higher `flask` version could bring unexpected result.

```
pip install flask=0.12.2
```
or 
```
conda install flask=0.12.2
```

### Run

1. Run CNN_model_for_occupancy.py to start training model. Make sure you find file `car1.h5` after training process.

2. Run `main.py` to start server. Your API should be available on http://127.0.0.1:8081/