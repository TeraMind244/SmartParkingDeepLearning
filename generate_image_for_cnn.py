
# %%

import image_utils as iu

# %%
# ### Initial value

pt_array = [[255, 24],
            [1080, 24],
            [1080, 700],
            [255, 700]]

height_scale = 720

# %%

lot_images = iu.get_all_image('test_images/*.jpg')
img_count = 0

for image in lot_images:
    resized_image = iu.resize(image, height_scale)

    white_yellow_image = iu.select_rgb_white_yellow(resized_image)
    gray_image = iu.convert_gray_scale(white_yellow_image)
    edge_image = iu.detect_edges(gray_image)
    #    show_image(edge_image)
    roi_image = iu.select_region(edge_image, pt_array)
    #    show_image(roi_image)
    lines = iu.hough_lines(roi_image)

    line_image, cleaned = iu.draw_lines(resized_image, lines)
    #    show_image(line_image)
    blocked_image, rects = iu.identify_blocks(resized_image, cleaned)
#    #    show_image(blocked_image)
    parking_slot_image, spot_dict = iu.draw_parking(resized_image, rects)
    iu.save_images_for_cnn(resized_image, spot_dict=spot_dict, img_count=img_count)
    img_count += 1
