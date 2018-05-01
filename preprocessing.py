import numpy as np
import pywt
import cv2
import os


def resize(mat):
    mat = cv2.resize(mat, (256, 256))
    return mat


def wavelet(mat):
    mat = np.float32(mat)
    m, (n, o, p) = pywt.dwt2(mat, "haar")
    m = np.uint8(m)
    (m, n, o, p) = (np.uint8(e) for e in (m, n, o, p))
    return m, (n, o, p)


def wvt(img):
    b, g, r = cv2.split(img)

    b, b_p = wavelet(b)
    g, g_p = wavelet(g)
    r, r_p = wavelet(r)
    img2 = cv2.merge([b, g, r])
    p = ([cv2.merge([b_p[i], g_p[i], r_p[i]]) for i in range(0, len(b_p))])

    return img2, p


def process_image(mat):
    mat = resize(mat)
    im, _ = wvt(mat)
    img, p = wvt(im)
    return [img, p[0], p[1], p[2]]


def script():
    cats_path = os.listdir("cats/")
    dogs_path = os.listdir("dogs/")
    cats_imgs = [cv2.imread("cats/" + cats_path[i]) for i in range(0, len(cats_path))]
    dogs_imgs = [cv2.imread("dogs/" + dogs_path[i]) for i in range(0, len(dogs_path))]
    cats_imgs_haar = [process_image(e) for e in cats_imgs]
    dogs_imgs_haar = [process_image(e) for e in dogs_imgs]
    return cats_imgs_haar, dogs_imgs_haar


cats, dogs = script()
