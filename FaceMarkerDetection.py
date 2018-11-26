# import os
import dlib
# import glob
import cv2
import numpy as np


def face_marker_detection(file_path, predictor_path):


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    img = dlib.load_rgb_image(file_path)

    #win = dlib.image_window()
    #win.clear_overlay()
    #win.set_image(img)

    # Check number of faces detected.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)   # Get the landmarks/parts for the face in box d
        #win.add_overlay(shape)     # Draw the face landmarks on the screen.
    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()
    return shape

def head_pose_estimator(shape, file_path):
    # Read Image
    im = cv2.imread(file_path);
    size = im.shape

        # 2D image points. If you change the image, you need to change vector

    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # nose tip
        (shape.part(8).x, shape.part(8).y),  # chin
        (shape.part(36).x, shape.part(36).y),  # left eye outer tip
        (shape.part(45).x, shape.part(45).y),  # right eye outer tip
        (shape.part(48).x, shape.part(48).y),  # mouth left outer tip
        (shape.part(54).x, shape.part(54).y)  # mouth right outer tip
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    print "Camera Matrix :\n {0}".format(camera_matrix)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs)

    print "Rotation Vector:\n {0}".format(rotation_vector)
    print "Translation Vector:\n {0}".format(translation_vector)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector,
                                                     camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255, 0, 0), 2)

    # Display image
    cv2.imshow("Output", im)
    cv2.waitKey(0)


if __name__ == '__main__':
    predictor_path = "Predictor/shape_predictor_68_face_landmarks.dat"
    file_path = "Images/Florence.jpg"
    # Iterate through folder:
    #for f in glob.glob(os.path.join(faces_folder_path, "straight.jpg")):
    #print("Processing file: {}".format(f))
    shape = face_marker_detection(file_path, predictor_path)
    head_pose_estimator(shape, file_path)