import cv2


def read_image_frames(file_path):
    vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
def get_live_video():
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Not Working ???
def capture_live_video(save_file_path):
    cap = cv2.VideoCapture(0)  # Capture video from camera

    # Get the width and height of frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                break
        else:
            break

    # Release everything if job is finished
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    FILE_PATH = "videos"
    SAVE_FILE_PATH = "videos/video_capture.mp4"
    #get_live_video()
    capture_live_video(SAVE_FILE_PATH)
    # read_image_frames(FILE_PATH)
