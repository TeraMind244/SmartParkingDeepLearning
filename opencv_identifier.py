
# %%
from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os
import glob
import numpy as np
#import time
# Imports for making predictions
from keras.models import load_model

# %% 
# ### Initial value

cleaned = {}

class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'

pt_array = [[276, 22],
            [1060, 15],
            [1111, 680],
            [254, 692]]

cwd = os.getcwd()

model = load_model('car1.h5')

#with keras.backend.get_session().graph.as_default():
#    model = load_model("sunshade/neural_network/model.h5")

# %%
def resize(image, v_height):
    height, width, depth = image.shape
    scale = v_height / height
    (newX, newY) = (width*scale, height*scale)
    return cv2.resize(image, (int(newX), int(newY)))

# %%

def show_image(image, cmap=None):
    plt.figure(figsize=(15, 12))
    plt.subplot(1, 1, 1)
    # use gray scale color map if there is only one channel
    cmap = 'gray' if len(image.shape) == 2 else cmap
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

# %%

def select_rgb_white_yellow(image):
    # white color mask
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


# %%
def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# %%
def detect_edges(image, low_threshold=20, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

# %%
def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        # in case, the input image has a channel dimension
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask)

def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # rows, cols = image.shape[:2]
    
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([pt_array], dtype=np.int32)
    return filter_region(image, vertices)

# %% [markdown]
# ### Hough line transform

# %%

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10,
                           threshold=10, minLineLength=30, maxLineGap=40)


# %%
def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image)  # don't want to modify the original
    global cleaned
    cleaned = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2-y1) <= 5 and abs(x2-x1) >= 50 and abs(x2-x1) <= 1000:
                cleaned.append((x1, y1, x2, y2))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    print(" No. lines detected: ", len(cleaned))
    return image, cleaned


# %% [markdown]
# Identify rectangular blocks of parking

# %%

def identify_blocks(image, lines, make_copy=True):
    if make_copy:
        new_image = np.copy(image)
    # Step 1: Create a clean list of lines

    # Step 2: Sort cleaned by x1 position
    import operator
    global sorted_lines, cleaned

    sorted_lines = sorted(cleaned, key=operator.itemgetter(0, 1))

    # Step 3: Find clusters of x1 close together - clust_dist apart
    clusters = {}
    dIndex = 0
    clus_dist = 55

    for i in range(len(sorted_lines) - 1):
        distance = abs(sorted_lines[i+1][0] - sorted_lines[i][0])
        if distance <= clus_dist:
            if not dIndex in clusters.keys():
                clusters[dIndex] = []
            clusters[dIndex].append(sorted_lines[i])
            clusters[dIndex].append(sorted_lines[i + 1])
        else:
            dIndex += 1

    # Step 4: Identify coordinates of rectangle around this cluster
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))
        if len(cleaned) > 5:
            cleaned = sorted(cleaned, key=lambda tup: tup[1])
            avg_y1 = cleaned[0][1]
            avg_y2 = cleaned[-1][1]
            cleaned = sorted(cleaned, key=lambda tup: tup[0])
            max_x1 = cleaned[0][0]
            cleaned = sorted(cleaned, key=lambda tup: tup[2])
            min_x2 = cleaned[-1][2]
            rects[i] = (max_x1, avg_y1, min_x2, avg_y2)
            i += 1

    print("Num Parking Lanes: ", len(rects)*2-2)
    # Step 5: Draw the rectangles on the image
    buff = -5
    for key in rects:
        tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
        cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
    return new_image, rects

# %%
def int_to_letter(i):
    lane = ''
    if i > 0:
        while True:
            mod = (i - 1) % 26
            lane = chr(mod + 65) + lane
            i = (i - mod + 1) // 26
            if i <= 0:
                break
    return lane

# %% [markdown]
# ### Identify each spot and count num of parking spaces
# %% [markdown]
# Next step-
# 1. Based on width of each parking line segment into individual spots
# 2. draw a visualization of all parking spaces

# %%

def draw_parking(image, rects, make_copy=True, 
                 color=[255, 0, 0], thickness=2, save=True):
    if make_copy:
        new_image = np.copy(image)
    gap = 65
    spot_dict = {}  # maps each parking ID to its coords
    tot_spots = 0
    
    lane_num = 0
    
    for key in rects:
        lane_num += 1
        lane1 = int_to_letter(lane_num)
        lane2 = int_to_letter(lane_num + 1)
        # Horizontal lines
        tup = rects[key]
        x1 = int(tup[0])
        x2 = int(tup[2])
        y1 = int(tup[1])
        y2 = int(tup[3])
#        gap = int(abs(y2-y1)//10)
        
        cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        num_splits = int(abs(y2-y1)//gap)
        for i in range(0, num_splits+1):
            y = int(y1 + i*gap)
            cv2.line(new_image, (x1, y), (x2, y), color, thickness)
        if key > 0 and key < len(rects) - 1:
            # draw vertical lines
            x = int((x1 + x2)/2)
            cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            
        # Add up spots in this lane
        if key == 0 or key == (len(rects) - 1):
            tot_spots += num_splits
        else:
            tot_spots += 2*num_splits

        # Dictionary of spot positions
        if key == 0 or key == (len(rects) - 1):
            for i in range(0, num_splits):
                y = int(y1 + i*gap)
                spot_dict[(x1, y, x2, y+gap)] = {'row': i + 1,
                                                  'lane': lane1}
        else:
            for i in range(0, num_splits):
                y = int(y1 + i*gap)
                x = int((x1 + x2)/2)
                spot_dict[(x1, y, x, y+gap)] = {'row': i + 1,
                                                  'lane': lane1}
                spot_dict[(x, y, x2, y+gap)] = {'row': i + 1,
                                                  'lane': lane2}
            lane_num += 1

    print("total parking spaces: ", tot_spots)
    if save:
        filename = 'with_parking.jpg'
        cv2.imwrite(filename, new_image)
    return new_image, spot_dict

# %%

def assign_spots_map(image, spot_dict, make_copy = True, color=[255, 0, 0], thickness=2):
    if make_copy:
        new_image = np.copy(image)
    for spot in spot_dict.keys():
        (x1, y1, x2, y2) = spot
        cv2.rectangle(new_image, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
    return new_image

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
    
    list_slots = []
    
    for spot in spot_dict.keys():
        all_spots += 1
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        #crop this image
        spot_img = image[y1:y2, x1:x2]
        spot_img = cv2.resize(spot_img, (48, 48))

        label = make_prediction(spot_img)
        
        slot = {'row': spot_dict[spot]['row'],
                'lane': spot_dict[spot]['lane'],
                'status': label}
        list_slots.append(slot)
        
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

    return new_image, list_slots, cnt_empty, all_spots

def get_first_image(path):
    test_images = [plt.imread(path) for path in glob.glob(path)]
    return test_images[0]

def identify_parking_spot(image):
#    lot_image = image
    lot_image = get_first_image('test_images/*.jpg')
#    print(lot_image)
#    lot_image = image

    #Resize and blur
    #kernel_size = 3
    #image = cv2.bilateralFilter(image, kernel_size, kernel_size * 2, kernel_size / 2)
    resized_image = resize(lot_image, 720)
    
    white_yellow_image = select_rgb_white_yellow(resized_image)
    gray_image = convert_gray_scale(white_yellow_image)
    edge_image = detect_edges(gray_image)
    
    roi_image = select_region(edge_image)
    
    lines = hough_lines(roi_image)
    
    line_image, cleaned = draw_lines(resized_image, lines)
    
    blocked_image, rects = identify_blocks(resized_image, lines)
    
    parking_slot_image, spot_dict = draw_parking(resized_image, rects)
    global final_spot_dict
    final_spot_dict = spot_dict

#    marked_spot_image = assign_spots_map(lot_image, spot_dict=final_spot_dict)
    predicted_image, slots, cnt_empty, all_spots = predict_on_image(resized_image, spot_dict=final_spot_dict)
    
    return predicted_image, cnt_empty, all_spots
    
#show_image(identify_parking_spot(get_first_image('test_images/*.jpg')))
    
# %%

#def test_return_image(image):
#    spot_dict = {(267, 30, 410, 95): {'row': 1, 'lane': 'A'}, 
#                 (267, 95, 410, 160): {'row': 2, 'lane': 'A'}, 
#                 (267, 160, 410, 225): {'row': 3, 'lane': 'A'}, 
#                 (267, 225, 410, 290): {'row': 4, 'lane': 'A'}, 
#                 (267, 290, 410, 355): {'row': 5, 'lane': 'A'}, 
#                 (267, 355, 410, 420): {'row': 6, 'lane': 'A'}, 
#                 (267, 420, 410, 485): {'row': 7, 'lane': 'A'}, 
#                 (267, 485, 410, 550): {'row': 8, 'lane': 'A'}, 
#                 (267, 550, 410, 615): {'row': 9, 'lane': 'A'}, 
#                 (267, 615, 410, 680): {'row': 10, 'lane': 'A'}, 
#                 (534, 31, 671, 96): {'row': 1, 'lane': 'B'}, 
#                 (671, 31, 808, 96): {'row': 1, 'lane': 'C'}, 
#                 (534, 96, 671, 161): {'row': 2, 'lane': 'B'}, 
#                 (671, 96, 808, 161): {'row': 2, 'lane': 'C'}, 
#                 (534, 161, 671, 226): {'row': 3, 'lane': 'B'}, 
#                 (671, 161, 808, 226): {'row': 3, 'lane': 'C'}, 
#                 (534, 226, 671, 291): {'row': 4, 'lane': 'B'}, 
#                 (671, 226, 808, 291): {'row': 4, 'lane': 'C'}, 
#                 (534, 291, 671, 356): {'row': 5, 'lane': 'B'}, 
#                 (671, 291, 808, 356): {'row': 5, 'lane': 'C'}, 
#                 (534, 356, 671, 421): {'row': 6, 'lane': 'B'}, 
#                 (671, 356, 808, 421): {'row': 6, 'lane': 'C'}, 
#                 (534, 421, 671, 486): {'row': 7, 'lane': 'B'}, 
#                 (671, 421, 808, 486): {'row': 7, 'lane': 'C'}, 
#                 (534, 486, 671, 551): {'row': 8, 'lane': 'B'}, 
#                 (671, 486, 808, 551): {'row': 8, 'lane': 'C'}, 
#                 (534, 551, 671, 616): {'row': 9, 'lane': 'B'}, 
#                 (671, 551, 808, 616): {'row': 9, 'lane': 'C'}, 
#                 (534, 616, 671, 681): {'row': 10, 'lane': 'B'}, 
#                 (671, 616, 808, 681): {'row': 10, 'lane': 'C'}, 
#                 (935, 33, 1076, 98): {'row': 1, 'lane': 'D'}, 
#                 (935, 98, 1076, 163): {'row': 2, 'lane': 'D'}, 
#                 (935, 163, 1076, 228): {'row': 3, 'lane': 'D'}, 
#                 (935, 228, 1076, 293): {'row': 4, 'lane': 'D'}, 
#                 (935, 293, 1076, 358): {'row': 5, 'lane': 'D'}, 
#                 (935, 358, 1076, 423): {'row': 6, 'lane': 'D'}, 
#                 (935, 423, 1076, 488): {'row': 7, 'lane': 'D'}, 
#                 (935, 488, 1076, 553): {'row': 8, 'lane': 'D'}, 
#                 (935, 553, 1076, 618): {'row': 9, 'lane': 'D'}}
#    
#    list_slots = []
#    all_spots = 0
#    cnt_empty = 0
#    new_image = np.copy(image)
#    overlay = np.copy(image)
#    color=[0, 255, 0]
#    alpha=0.5
#    
#    for spot in spot_dict.keys():
#        all_spots += 1
#        (x1, y1, x2, y2) = spot
#        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
#        #crop this image
#        spot_img = image[y1:y2, x1:x2]
#        spot_img = cv2.resize(spot_img, (48, 48))
#
#        label = make_prediction(spot_img)
#        
#        slot = {'row': spot_dict[spot]['row'],
#                'lane': spot_dict[spot]['lane'],
#                'status': label}
#        list_slots.append(slot)
#        
##        print(label)
#        if label == 'empty':
#            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), 
#                          color, -1)
#            cnt_empty += 1
#            
#    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)
#    return overlay, all_spots, cnt_empty

# %%
    
#lot_image = image
#lot_image = get_first_image('test_images/*.jpg')
##    print(lot_image)
##    lot_image = image
#
#    #Resize and blur
#    #kernel_size = 3
#    #image = cv2.bilateralFilter(image, kernel_size, kernel_size * 2, kernel_size / 2)
#resized_image = resize(lot_image, 720)
#
#white_yellow_image = select_rgb_white_yellow(resized_image)
#gray_image = convert_gray_scale(white_yellow_image)
#edge_image = detect_edges(gray_image)
#
#roi_image = select_region(edge_image)
#
#lines = hough_lines(roi_image)
#
#line_image, cleaned = draw_lines(resized_image, lines)
#
#blocked_image, rects = identify_blocks(resized_image, lines)
#
#parking_slot_image, spot_dict = draw_parking(resized_image, rects)
#global final_spot_dict
#final_spot_dict = spot_dict
#
##    marked_spot_image = assign_spots_map(lot_image, spot_dict=final_spot_dict)
#predicted_image, slots = predict_on_image(resized_image, spot_dict=final_spot_dict)

#return slots

# %%
    
#def get_data(spot_dict):
#    all_spots = 0
#    cnt_empty = 0
#    for spot in spot_dict.keys():
#        all_spots += 1
#        (x1, y1, x2, y2) = spot
#        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
#        #crop this image
#        spot_img = image[y1:y2, x1:x2]
#        spot_img = cv2.resize(spot_img, (48, 48))
#
#        label = make_prediction(spot_img)
#        
##        print(label)
#        if label == 'empty':
#            cnt_empty += 1
#    return all_spots, cnt_empty
