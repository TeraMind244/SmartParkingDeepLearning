
# %%
#from __future__ import division
import cv2
import numpy as np
# Imports for making predictions
from keras.models import load_model
import image_utils as iu
import read_data as rd

# %% 
# ### Initial value

class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'

data = rd.get_data()

model = load_model('car1.h5')

# %%


def make_prediction(spot_image):
    #Rescale image
    img = spot_image/255.

    #Convert to a 4D tensor
    spot_image = np.expand_dims(img, axis=0)
    #print(image.shape)

    # make predictions on the preloaded model
    class_predicted = model.predict(spot_image)
    inID = np.argmax(class_predicted[0])
    label = class_dictionary[inID]
    return label


# %%
    

def predict_on_image(image, spot_dict, color=[0, 255, 0],
                     alpha=0.5):
    spot_list = []
    count = 0
    
    for spot in spot_dict.keys():
        count += 1
        position = spot_dict[spot]
#        all_spots += 1
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        #crop this image
        spot_img = image[y1:y2, x1:x2]
        spot_img = cv2.resize(spot_img, (48, 48))

        label = make_prediction(spot_img)
        
        slot = {'row': position['row'],
                'lane': position['lane'],
                'status': label,
                'position': spot}
        spot_list.append(slot)

    return spot_list


# %%
    

def detect_parking_image(image, camId, debug=False):
    test_image = image
    
    cam_data = next(filter(lambda cam: cam['cam_id'] == camId, data['cam_list']), {})
    if not cam_data:
        return [], [], 1.0
    
    roi_array = cam_data['roi_array']
    pt_array = cam_data['pt_array']
    kernel_size = cam_data['kernel_size']
    gap = cam_data['gap']
    lane_list = cam_data['lane_list']
    
    if debug:
        test_image = iu.get_first_image('webcam_test/test_frame.jpg')
        iu.show_image(test_image)
    
    roi_image = iu.select_region(test_image, roi_array)
    if debug:
        iu.show_image(roi_image)
    
    rotated_image, transform_matrix = iu.four_point_transform(roi_image, pt_array)
    if debug:
        iu.show_image(rotated_image)
    
    resized_image, scale = iu.resize(rotated_image, 360)
    if debug:
        iu.show_image(resized_image)
    
    blurred_image = iu.blur(resized_image, kernel_size=kernel_size)
    if debug:
        iu.show_image(blurred_image)
    
    edge_image = iu.detect_edges(blurred_image)
    if debug:
        iu.show_image(edge_image)
    
    lines = iu.hough_lines(edge_image)
    
    line_image, cleaned = iu.draw_lines(resized_image, lines)
    if debug:
        iu.show_image(line_image)
    
    blocked_image, rects = iu.identify_blocks(resized_image, cleaned)
    if debug:
        iu.show_image(blocked_image)
        
    parking_slot_image, spot_dict = iu.draw_parking(resized_image, rects, gap=gap, lane_list=lane_list)
    if debug:
        iu.show_image(parking_slot_image)
    
    spot_list = predict_on_image(parking_slot_image, spot_dict)
    
    if debug:
        final_image = iu.reverse_image(test_image, transform_matrix, spot_list, scale)
        iu.show_image(final_image)
    
    return transform_matrix, spot_list, scale

#transform_matrix, spot_list, scale = detect_parking_image(None, 2, debug=True)
