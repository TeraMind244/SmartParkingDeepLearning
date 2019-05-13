#import numpy as np
import cv2
import image_utils as iu
import sys
import read_data as rd
#import opencv_identifier as opencv

# %%

data = rd.get_data()


# %%


def capture_video(source):
    cap = cv2.VideoCapture(source)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = None
    capturing = False

    while(True):
        ret, frame = cap.read()

        if ret == True:

            # Write the frame into the file 'output.avi'
            if capturing:
                out.write(frame)
                cv2.putText(frame, 'CAPTURING', (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'):
                break
            
            if keypress == ord('c'):
                if out == None:
                    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
                        'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
                if capturing: 
                    capturing = False
                else:
                    capturing = True
                    
        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objectsqq
    if out != None:
        out.release()
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


# %%

def show_video():
    cap = cv2.VideoCapture('outpy.avi')
    import time
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:

            # Display the resulting frame
            cv2.imshow('frame', frame)
            time.sleep(0.1)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objectsqq
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


# %% 

def write_a_random_frame():
    cap = cv2.VideoCapture('outpy.avi')
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count += 1
            if count == 20:
                filename = "webcam_test/test_frame.jpg"
                cv2.imwrite(filename, frame)
        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objectsqq
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


# %%
    
def write_random_frames():
    cap = cv2.VideoCapture('outpy.avi')
#    import time
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if count % 10 == 0:
                filename = "webcam_test/test_frame_%d.jpg" %count
#                print(filename)
                cv2.imwrite(filename, frame)
            count += 1
        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objectsqq
    cap.release()


# %%


def save_images_for_cnn(camId):
    images = iu.get_all_image('webcam_test/*.jpg')

    img_count = 0
    for image in images:
        test_image = image
    
        cam_data = next(filter(lambda cam: cam['cam_id'] == camId, data['cam_list']))
        roi_array = cam_data['roi_array']
        pt_array = cam_data['pt_array']
        kernel_size = cam_data['kernel_size']
        gap = cam_data['gap']
        lane_list = cam_data['lane_list']
        
        roi_image = iu.select_region(test_image, roi_array)
        
        rotated_image, transform_matrix = iu.four_point_transform(roi_image, pt_array)
        
        resized_image, scale = iu.resize(rotated_image, 360)
        
        blurred_image = iu.blur(resized_image, kernel_size=kernel_size)
        
        edge_image = iu.detect_edges(blurred_image)
        
        lines = iu.hough_lines(edge_image)
        
        line_image, cleaned = iu.draw_lines(resized_image, lines)
        
        blocked_image, rects = iu.identify_blocks(resized_image, cleaned)
            
        parking_slot_image, spot_dict = iu.draw_parking(resized_image, rects, gap=gap, lane_list=lane_list)
        
        iu.save_images_for_cnn(resized_image, spot_dict, folder_name='for_cnn/', img_count=img_count)
        
        img_count += 1


# %%


#def main():
#    # print command line arguments
#    mode = sys.argv[1]
#    if mode == "capture":
#        capture_video()
#        return
#    if mode == "show-video":
#        show_video()
#        return
#    if mode == "save-a-frame":
#        write_a_random_frame()
#        return
#    if mode == "save-frames":
#        write_random_frames()
#        return
#    if mode == "get-training-data":
#        save_images_for_cnn()
#        return
#    print("You provided wrong input! Please try again!")
#    
#    source = sys.argv[1]
#    capture_video(int(source))
#    write_a_random_frame()

#if __name__ == "__main__":
#    main()

    
# %%%
    
#cap1 = cv2.VideoCapture(1)
#cap2 = cv2.VideoCapture(2)
#
#while(True):
#    ret, frame1 = cap1.read()
#    ret, frame2 = cap2.read()
#
#    if ret == True:
#
#        # Display the resulting frame
#        cv2.imshow('frame1', frame1)
#        cv2.imshow('frame2', frame2)
#
#        # Press Q on keyboard to stop recording
#        keypress = cv2.waitKey(1) & 0xFF
#        if keypress == ord('q'):
#            break
#                
#    # Break the loop
#    else:
#        break
#
#cap1.release()
#cap2.release()
#
## Closes all the frames
#cv2.destroyAllWindows()
        

# %%
    
capture_video(2)
#show_video()
write_a_random_frame()
#write_random_frames()
#save_images_for_cnn(2)

