# This file defines the functions used to
# - Rotate images and crop them
# - Average the vertical intensity

import cv2
import numpy as np
import math
import re
import glob
from PIL import Image
import matplotlib.pyplot as plt

__DEBUG__ = True

def debug(s):
    global __DEBUG__
    if __DEBUG__:
        print(s)


def voltage_from_filepath(s):
    return float(re.search('(?<=_)([0-9]+[.0-9]*)(?=V_)', s).group(1))


def get_bool(prompt):
    while True:
        inpt = input(f"{prompt} (True/False): ")
        if inpt.lower() in ['true', 't']:
            return True
        elif inpt.lower() in ['false', 'f']:
            return False
        else:
            print("Invalid input, please try again")

def get_filename(file_path: str) -> str:
    filename = re.search('([^/]+$)', file_path).group(1)
    return filename


def average_vertical_intensity(image_filepath: str, save_path: str = None, show_plot: bool = False) -> np.array:
    """
    Given a file path, opens the image, and averages over the vertical axis
    To get intensities of the image. Then, optionally plots and saves it.
    """
    img = np.asarray(Image.open(image_filepath))
    filename = get_filename(image_filepath)
    if save_path == None:
        raise ValueError("Need a save path")

    if len(img.shape) == 3:
        img = np.sum(img, axis=2)

    mid = int(img.shape[0]/2)

    intensities = np.sum(img[mid-50:mid+50, :], axis=0)/100
    plt.scatter(list(range(img.shape[1])), intensities, s=0.3)
    plot_path = save_path + '/' + filename[:-4] + '_intensity.png'
    plt.savefig(plot_path)
    if show_plot:
        plt.show()
    else:
        plt.clf()

    intensities_file_path = save_path + '/' + filename[:-4] + '_intensity.npy'

    np.save(intensities_file_path, intensities)
    
    return intensities_file_path

def get_image_filepaths(path, element, line):
    file_list = glob.glob(f"{path}/{element}_{line}_*V.png")
    return file_list

def rotate_and_crop_images(image_filepaths: list[str], save_path: str):
    """
    Args: List of png image paths
    Does: Prompts user about rotating them until good. Then saves them.
    Returns: List of images, rotated and cropped
    """
    
    satisfied = False

    images = [cv2.imread(img) for img in image_filepaths]

    while not satisfied:
        
        rotation = float(input("Please input rotation (in degrees): "))
        
        rotated_images = [rotate_and_crop(img, rotation) for img in images]

        for img in rotated_images:
            img2 = img.copy()
            img2[:, 0::50] = 1
            cv2.imshow("rotated", img2)
            cv2.waitKey(0)

        satisfied = get_bool("Works?")

    save_paths = []

    for img, path in zip(rotated_images, image_filepaths):
        debug(f"{path=}")
        filename = get_filename(path)
        file_save_path = save_path + '/' + filename[:-4] + '_' + str(rotation) + '.png'
        cv2.imwrite(file_save_path, img)
        save_paths.append(file_save_path)

    return save_paths

def rotate_and_crop(image, angle):
    rotated_image = rotate_image(image, angle)
    w, h = largest_rotated_rect(image.shape[1], image.shape[0], angle)
    return crop_around_center(rotated_image, w, h)

def rotate_image(image, angle):

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    angle = math.radians(angle)

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


