#! /usr/bin/python
import cProfile
import cv2
import numpy as np
import pdb

def NNinterpol():
    M, N = 1024, 1024
    img = cv2.imread("../imgs/cameraman.tif", cv2.IMREAD_GRAYSCALE)
    R = img.shape[0]
    C = img.shape[1]
    j = np.floor(M/R)
    k = np.floor(N/C)
    repeat = np.ones((j, k))
    return np.array(np.kron(img, repeat), dtype=np.uint8)

if __name__ == '__main__':
    NNinterpol()
