import numpy as np
import FaceMarkerDetection as fmd
import glob
import os

def get_eye_pose(head_pose):
    eye_to_head = np.linalg.inv(head_pose)
    return eye_to_head


def get_eye_model_3d(x, y):
    """
    Using the calibration model return theta and phi as a function of x and y
    :param x: centre x coordinate of the eye ball in 2d
    :param y: centre y coordinate of the eye ball in 2d
    :return:
    """
    return NotImplemented
    

if __name__ == '__main__':
    FACES_FOLDER_PATH = "Images/Eye_Calibration"
    predictor_path = "Predictor/shape_predictor_68_face_landmarks.dat"

    # Iterate through folder:
    for f in glob.glob(os.path.join(FACES_FOLDER_PATH, "*.jpg")):
        # For the set of calibration images in this folder the eye to camera matrix is Identity
        print("Processing file: {}".format(f))
        shape = fmd.face_marker_detection(f, predictor_path)
        head_pose = fmd.head_pose_estimator(shape, f)
        eye_pose = get_eye_pose(head_pose)
        print (eye_pose)
        # TODO overlay normal onto image at eye centre.