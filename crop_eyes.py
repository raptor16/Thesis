import cv2
import numpy as np
import FaceMarkerDetection as fmd
import Rectangle
from matplotlib import pyplot as plt
import colour_splitter

# get coordinates for eyes using dlib


def get_eye_coordinates(file_path, predictor_path):
    shape = fmd.face_marker_detection(file_path, predictor_path)
    num_of_eye_markers = (5, 2)
    left_eye, right_eye = np.empty(num_of_eye_markers), np.empty(num_of_eye_markers)
    # left eye markers:  37, 38, 39, 40, 41, 42; right eye markers: 43, 44, 45, 46, 47, 48

    for i in range (num_of_eye_markers[0]):
        left_eye[i] = part2array(shape.part(36+i))
        right_eye[i] = part2array(shape.part(43 + i))

    return left_eye, right_eye


def part2array(part):
    shape = (2, 1)
    array = np.empty(shape)
    array[0] = part.x
    array[1] = part.y
    return np.transpose(array)


def get_boundng_box(eye):
    minXY = np.amin(eye, axis=0)
    maxXY =  np.amax(eye, axis=0)
    rect = Rectangle.Rectangle()
    rect.set_top_left(minXY)
    rect.set_bottom_right(maxXY)
    return rect


def crop_image(file_path, left_bb, right_bb):
    im = cv2.imread(file_path)

    xLeft = int(left_bb.get_rectange_coords()[0][0])
    yLeft = int(left_bb.get_rectange_coords()[0][1])
    print left_bb.get_rectange_coords()
    print xLeft, yLeft
    print right_bb.get_rectange_coords()
    croppedL = im[yLeft:yLeft + int(left_bb.get_height_pixels()), xLeft:xLeft + int(left_bb.get_width_pixels())]

    xRight = int(right_bb.get_rectange_coords()[0][0])
    yRight = int(right_bb.get_rectange_coords()[0][1])
    croppedR = im[yRight:yRight + int(right_bb.get_height_pixels()), xRight:xRight + int(right_bb.get_width_pixels())]

    plt.subplot(2, 1, 1), plt.imshow(croppedL), plt.title("Left Eye")
    plt.subplot(2, 1, 2), plt.imshow(croppedR), plt.title("Right Eye")
    plt.show()

    return croppedL, croppedR

def canny_edge_detector(croppedL, croppedR, thresh1=100, thresh2=200):
    edgesL = cv2.Canny(croppedL, 100, 200)
    edgesR = cv2.Canny(croppedR, 100, 200)
    plt.subplot(2, 2, 1), plt.imshow(croppedL), plt.title("Original Left")
    plt.subplot(2, 2, 2), plt.imshow(edgesL), plt.title("Edges Left")
    plt.subplot(2, 2, 3), plt.imshow(croppedR), plt.title("Original Right")
    plt.subplot(2, 2, 4), plt.imshow(edgesR), plt.title("Edges Right")
    plt.show()
    return edgesL, edgesR


if __name__ == '__main__':
    PREDICTOR_PATH = "Predictor/shape_predictor_68_face_landmarks.dat"
    FILE_PATH = "Images/ez.jpg"
    left_eye, right_eye = get_eye_coordinates(FILE_PATH, PREDICTOR_PATH)
    left_bb = get_boundng_box(left_eye)
    right_bb = get_boundng_box(right_eye)
    croppedL, croppedR = crop_image(FILE_PATH, left_bb, right_bb)

    edgesL, edgesR = canny_edge_detector(croppedL, croppedR)

    bl, gl, rl, br, gr, rr = colour_splitter.split_colours(croppedL, croppedR)
    canny_edge_detector(rl, rr)
