# %%
#from __future__ import division
import matplotlib.pyplot as plt
import cv2
#import os
import glob
import numpy as np

cleaned = {}

# %%


def show_image(image, cmap=None):
    plt.figure(figsize=(16, 9))
    plt.subplot(1, 1, 1)
    # use gray scale color map if there is only one channel
    cmap = 'gray' if len(image.shape) == 2 else cmap
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=cmap)


# %%


def resize(image, v_height):
    height, width, depth = image.shape
    scale = v_height / height
    (newX, newY) = (width*scale, height*scale)
    return cv2.resize(image, (int(newX), int(newY)))

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


def detect_edges(image, low_threshold=5, high_threshold=80):
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


def select_region(image, pt_array):
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
                           threshold=10, minLineLength=30, maxLineGap=30)


# %%
def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image)  # don't want to modify the original
    cleaned = []
    if lines is None:
        lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2-y1) <= 5 and abs(x2-x1) >= 35 and abs(x2-x1) <= 200:
                cleaned.append((x1, y1, x2, y2))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    print(" No. lines detected: ", len(cleaned))
    return image, cleaned


# %% [markdown]
# Identify rectangular blocks of parking

def identify_blocks(image, cleaned, make_copy=True):
    if make_copy:
        new_image = np.copy(image)
    # Step 1: Create a clean list of lines

    # Step 2: Sort cleaned by x1 position
    import operator
    # global sorted_lines, cleaned

    sorted_lines = sorted(cleaned, key=operator.itemgetter(0, 1))

    # Step 3: Find clusters of x1 close together - clust_dist apart
    clusters = {}
    dIndex = 0
    clust_dist = 55

    for i in range(len(sorted_lines) - 1):
        distance = abs(sorted_lines[i+1][0] - sorted_lines[i][0])
        if distance <= clust_dist:
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
        lines_in_clust = list(set(all_list))
        if len(lines_in_clust) > 5:
            lines_in_clust = sorted(lines_in_clust, key=lambda tup: tup[1])
            min_y1 = lines_in_clust[0][1]
            max_y2 = lines_in_clust[-1][1]
            lines_in_clust = sorted(lines_in_clust, key=lambda tup: tup[0])
            max_x1 = lines_in_clust[0][0]
            lines_in_clust = sorted(lines_in_clust, key=lambda tup: tup[2])
            min_x2 = lines_in_clust[-1][2]
            rects[i] = (max_x1, min_y1, min_x2, max_y2)
            i += 1

    print("Num Parking Lanes: ", len(rects)*2-2)
    # Step 5: Draw the rectangles on the image
    buff = -5
    for key in rects:
        tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
        cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 2)
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

# %%


def draw_parking(image, rects, gap=65, make_copy=True,
                 color=[255, 0, 0], thickness=2, save=False):
    if make_copy:
        new_image = np.copy(image)
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


def get_first_image(path):
    test_images = [plt.imread(path) for path in glob.glob(path)]
    return test_images[0]


# %%


def get_all_image(path):
    test_images = [plt.imread(path) for path in glob.glob(path)]
    return test_images


# %%
    

def save_img(filename, image, is_gray=False):
    if is_gray:
        cv2.imwrite(filename, image)
    else:
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# %%


def save_images_for_cnn(image, spot_dict, folder_name='for_cnn/', img_count=0):
    for spot in spot_dict.keys():
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        spot_image = image[y1:y2, x1:x2]
        spot_image = cv2.resize(spot_image, (48, 48))
        spot_id = str(spot_dict[spot]['lane']) + str(spot_dict[spot]['row'])

        filename = str(img_count) + '_spot' + spot_id + '.jpg'
        print(folder_name + filename, (x1, x2, y1, y2))

        save_img(folder_name + filename, spot_image)
        
# %%

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    pts = np.array(pts, dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
    return rect        


# %% 
        
        
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
    		[0, 0],
    		[maxWidth - 1, 0],
    		[maxWidth - 1, maxHeight - 1],
    		[0, maxHeight - 1]
        ], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
    return warped


# %%
    

def rotate(image, angle):
    
    (h, w) = image.shape[:2]
    
    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, -angle, scale)
    rotated = cv2.warpAffine(image, M, (h, w))
    return rotated

# %%


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

