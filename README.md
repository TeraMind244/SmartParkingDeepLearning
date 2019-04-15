# Smart Parking

### Contents:

1. `test_images` - Sample image to run and test the code on

2. `train_data` - Training data for CNN model

3. `for_cnn` - Directory to keep generated images for cnn. You have to manually label these images by separating them into sub-folders as structured below

```
.
├── for_cnn
└── test_images
    ├──train
    |   ├──empty
    |   └──occupied
    └── test
        ├──empty
        └──occupied
```

For the best training result, you should separate them under ratio `8:2` for `train:test`

4. `camera.py` - Python class for getting frames from your webcam

Change
```
self.video = cv2.VideoCapture(1)
```
to
```
self.video = cv2.VideoCapture(0)
```
if you don't have external webcam

5. `CNN_model_for_occupancy.py` - Python script for starting your model training process

6. `opencv_identifier.py` - Python script for parking spot detection

7. `main.py` - Python script using Flask to provide API

8. `config.json` - Dynamic config file

9. `read_data.py` - Python script to read data from `config.json`

10. `webcam_utils.py` - Python utils script that help you capturing video from webcam, save image for cnn model, etc

11. `webcam_test` - Directory to save images from running `webcam_utils.py`

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

### Usage of `webcam_utils.py`
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

1. Prepare training data with `webcam_utils.py`. Your training data should available at `test_images/` with structured above. However, we have prepared some sample data for you in `test_images/`. Therefore, you may skip this step.

2. Run `CNN_model_for_occupancy.py` to start training model. Make sure you find file `car1.h5` after training process.

3. Run `main.py` to start server. Your API should be available on http://0.0.0.0:8081/
