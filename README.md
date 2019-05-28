# Smart Parking

### Note:

Please note that this repository is build based on [Priya Dwivedi](https://github.com/priya-dwivedi)'s project on GitHub.
You can find the her article [here](https://towardsdatascience.com/find-where-to-park-in-real-time-using-opencv-and-tensorflow-4307a4c3da03) and her base source code [here](https://github.com/priya-dwivedi/Deep-Learning/tree/master/parking_spots_detector).

### Contents:

1. `webcam_test/` - Sample image to run and test the code on

2. `train_data/` - Training data for CNN model. Sample data included

3. `for_cnn/` - Directory to keep generated images for cnn. You have to manually label these images by separating them into sub-folders in side `train_data/` as structured below. Please note that `for_cnn/` is just only a directory to keep temporary images. It has no use in main process.

```
.
├── for_cnn
└── train_data
    ├──train
    |   ├──empty
    |   └──occupied
    └── test
        ├──empty
        └──occupied
```

For the best training result, you should separate them under ratio `8:2` for `train:test`

4. `templates` - Sample template to test API provided by Flask

5. `camera.py` - Python class for getting frames from your webcam

Change
```
self.video = cv2.VideoCapture(1)
```
to
```
self.video = cv2.VideoCapture(0)
or
self.video = cv2.VideoCapture('your_video.avi')
```
if you don't have external webcam

6. `CNN_model_for_occupancy.py` - Python script for starting your model training process. In the first time you run the script, it may take a long time for downloading the model. Please note that I am using Priya Dwivedi's model and it works well on my data.

7. `opencv_identifier.py` - Python script for parking spot detection

8. `main.py` - Python script using Flask to provide API

9. `config.json` - Config file for dynamic data

10. `read_data.py` - Python script to read data from `config.json`. This file is imported in other Python files. You do not need to run this file.

11. `webcam_utils.py` - Python utils script that help you capturing video from webcam and save image for cnn model, etc. User's manual can be found below.

12. `image_utils` - Python utils script that implement `Opencv` framework's algorithm.

### Installation

1. You would need Python 2.7 or 3.6 [here](https://www.python.org/downloads/release/python-368/)

2. Please follow [this instruction](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) to install `tensorflow` or `tensorflow-gpu`, depends on your hardware.

3. Install `opencv2`
```
pip install opencv
or 
conda install opencv
```

4. Install `flask` version `0.12.2`. Please note that higher `flask` version could bring unexpected result.

```
pip install flask=0.12.2
or 
conda install flask=0.12.2
```

### User's manual for `webcam_utils.py`
Run
```
python webcam_utils.py --mode
```
Where `--mode`:
1. `capture` - Start a video capturing instance. Press `q` to stop. Press `c` to toggle capturing mode. Video captured is save as `outpy.avi`

2. `show-video` - Show the video you captured as `outpy.avi`

3. `save-a-frame` - Read `outpy.avi` and save a random frame as `webcam_test/test_frame.jpg`

4. `save-frames` - Read `outpy.avi` and save random frames into `webcam_test/`

5. `get-training-data` - Read all images in `webcam_test/` and generate slot images into `for_cnn/`

### Run

1. Prepare training data with `webcam_utils.py`. Your prepared data will be in `for_cnn/`. However, please note that training data should be available in `train_data/` with structured above. We have prepared some sample data for you in `train_data/`. Therefore, you may skip this step.

2. Run `CNN_model_for_occupancy.py` to start training model. Make sure you find file `car1.h5` after training process.

3. Run `main.py` to start server. Your API should be available on http://0.0.0.0:8081/
