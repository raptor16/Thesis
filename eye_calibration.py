import numpy as np
import FaceMarkerDetection as fmd
import glob
import os
import cv2

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


def get_rotation_translation_vector(transformation_matrix):
    vec_shape = (3,)
    t_vec = np.empty(vec_shape)

    r_vec, _ = cv2.Rodrigues(transformation_matrix)
    for i in range(t_vec.shape[0]):
        t_vec[i] = transformation_matrix[i][3]
    return r_vec, t_vec


def get_camera_info(file_path, im):
    size = im.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    return camera_matrix, dist_coeffs


def overlay_normal(file_path, shape):
    im = cv2.imread(file_path);
    rotation_vector, translation_vector = get_rotation_translation_vector(eye_pose)
    camera_matrix, dist_coeffs = get_camera_info(file_path, im)
    (eye_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                    translation_vector,
                                                    camera_matrix, dist_coeffs)
    # centre of two eyes coordinates
    centre_eyes_x = (shape.part(36).x + shape.part(45).x) / 2
    centre_eyes_y = (shape.part(36).y + shape.part(45).y) / 2
    p1 = (int(centre_eyes_x), int(centre_eyes_y))
    p2 = (int(eye_end_point2D[0][0][0]), int(eye_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255, 0, 0), 2)

    # Display image
    cv2.imshow("Eye Pose", im)
    cv2.waitKey(0)


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
        overlay_normal(f, shape)
        # TODO overlay normal onto image at eye centre.