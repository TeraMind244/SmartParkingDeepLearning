
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

pt_array = data['pt_array']

height_scale = data['height_scale']
gap = data['gap']

#cwd = os.getcwd()

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

def predict_on_image(image, spot_dict, make_copy=True, color=[0, 255, 0],
                     alpha=0.5, save=False):
    if make_copy:
        new_image = np.copy(image)
        overlay = np.copy(image)
    cnt_empty = 0
    all_spots = 0
    
    spot_list = []
    
    for spot in spot_dict.keys():
        position = spot_dict[spot]
        all_spots += 1
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
        
#        print(label)
        if label == 'empty':
            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color, -1)
            cnt_empty += 1

    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

    cv2.putText(new_image, "Available: %d spots" %cnt_empty, (30, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (255, 255, 255), 2)

    cv2.putText(new_image, "Total: %d spots" %all_spots, (30, 125),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (255, 255, 255), 2)

    if save:
        filename = 'with_marking.jpg'
        cv2.imwrite(filename, new_image)

    return new_image, spot_list


# %%

def identify_parking_spot(image):
#    global lot_image, resized_image
    lot_image = image
#    lot_image = get_first_image('test_images/*.jpg')
#    print(lot_image)
#    lot_image = image

    #Resize and blur
    #kernel_size = 3
    #image = cv2.bilateralFilter(image, kernel_size, kernel_size * 2, kernel_size / 2)
    resized_image = iu.resize(lot_image, height_scale)
    
    white_yellow_image = iu.select_rgb_white_yellow(resized_image)
    gray_image = iu.convert_gray_scale(white_yellow_image)
#    show_image(gray_image)
    edge_image = iu.detect_edges(gray_image)
#    show_image(edge_image)
    roi_image = iu.select_region(edge_image, pt_array)
#    show_image(roi_image)
    lines = iu.hough_lines(roi_image)
    
    line_image, cleaned = iu.draw_lines(resized_image, lines)
#    show_image(line_image)
    blocked_image, rects = iu.identify_blocks(resized_image, cleaned)
#    show_image(blocked_image)
    parking_slot_image, spot_dict = iu.draw_parking(resized_image, rects, gap=gap)
#    iu.show_image(parking_slot_image)

#    marked_spot_image = assign_spots_map(lot_image, spot_dict=final_spot_dict)
    final_image, list_slots = predict_on_image(resized_image, spot_dict=spot_dict)
#    iu.show_image(final_image)
    return list_slots


# %%
    
#lot_image = iu.get_first_image('webcam_test/*.jpg')
##    print(lot_image)
##    lot_image = image
#
##Resize and blur
##kernel_size = 3
##image = cv2.bilateralFilter(image, kernel_size, kernel_size * 2, kernel_size / 2)
#resized_image = iu.resize(lot_image, height_scale)
#iu.show_image(resized_image)
#white_yellow_image = iu.select_rgb_white_yellow(resized_image)
#gray_image = iu.convert_gray_scale(white_yellow_image)
##iu.show_image(gray_image)
#edge_image = iu.detect_edges(gray_image)
##iu.show_image(edge_image)
#roi_image = iu.select_region(edge_image, pt_array)
#iu.show_image(roi_image)
#lines = iu.hough_lines(roi_image)
#
#line_image, cleaned = iu.draw_lines(resized_image, lines)
#iu.show_image(line_image)
#blocked_image, rects = iu.identify_blocks(resized_image, cleaned)
#iu.show_image(blocked_image)
#parking_slot_image, spot_dict = iu.draw_parking(resized_image, rects, gap=gap)
#iu.show_image(parking_slot_image)
##    global final_spot_dict
##    final_spot_dict = spot_dict
#
##    marked_spot_image = assign_spots_map(lot_image, spot_dict=final_spot_dict)
#final_image, list_slots = predict_on_image(resized_image, spot_dict=spot_dict)
#iu.show_image(final_image)


