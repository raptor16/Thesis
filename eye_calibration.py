import numpy as np
import FaceMarkerDetection as fmd
import glob
import os

def get_eye_pose(head_pose):
    eye_to_head = np.linalg.inv(head_pose)
    return eye_to_head


def get_eye_model():
    

if __name__ == '__main__':
    FACES_FOLDER_PATH = "Images/Eye_Calibration"
    # Iterate through folder:
    for f in glob.glob(os.path.join(FACES_FOLDER_PATH, "*.jpg")):
        # For the set of calibration images in this folder the eye to camera matrix is Identity
        print("Processing file: {}".format(f))
        shape = fmd.face_marker_detection()
        head_pose = fmd.hp(shape, f)
        get_eye_pose(head_pose)