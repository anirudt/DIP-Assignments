import cv2
import numpy as np

def otsu_range(img):
    """ Performs Otsu's thresholding algorithm """
    nbins = 256
    hist = np.bincount(img)
    p = hist / nbins
    sigma_b = np.zeros(256)

    for i in xrange(nbins):
        q_l = np.sum(p[0:i])
        q_h = np.sum(p[i+1])
        mu_l = np.sum(p[0:i] * np.arange(0,i+1).T)/q_l
        mu_h = np.sum(p[i+1:] * np.arange(i+1,256).T)/q_h
        sigma_b[i] = q_l * q_h * (mu_h - mu_l)**2

    return sigma_b.argmax()


if __name__ == "__main__":
    img = cv2.imread("cameraman.tif")
