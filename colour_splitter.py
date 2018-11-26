import cv2
from matplotlib import pyplot as plt


def split_colours(filename):
    img = cv2.imread(filename)
    b, g, r = cv2.split(img)
    plt.subplot(3, 1, 1), plt.imshow(b), plt.title("blue channel")
    plt.subplot(3, 1, 2), plt.imshow(g), plt.title("green channel")
    plt.subplot(3, 1, 3), plt.imshow(r), plt.title("red channel")
    plt.show()

def split_colours(imgL, imgR):
    bl, gl, rl = cv2.split(imgL)
    plt.subplot(3, 2, 1), plt.imshow(bl), plt.title("L blue channel")
    plt.subplot(3, 2, 3), plt.imshow(gl), plt.title("L green channel")
    plt.subplot(3, 2, 5), plt.imshow(rl), plt.title("L red channel")
    br, gr, rr = cv2.split(imgR)
    plt.subplot(3, 2, 2), plt.imshow(br), plt.title("R channel")
    plt.subplot(3, 2, 4), plt.imshow(gr), plt.title("R channel")
    plt.subplot(3, 2, 6), plt.imshow(rr), plt.title("R channel")
    plt.show()
    return bl, gl, rl, br, gr, rr

if __name__ == '__main__':
    # for f in glob.glob(os.path.join(faces_folder_path, "straight.jpg")):
    FILENAME = "Images/Florence.jpg"
    split_colours(FILENAME)