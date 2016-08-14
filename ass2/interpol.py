#! /usr/bin/python
import cProfile
import cv2
import numpy as np
import pdb

def NNinterpol(img, M, N):
    R = img.shape[0]
    C = img.shape[1]
    j = np.floor(M/R)
    k = np.floor(N/C)
    repeat = np.ones((j, k))
    img = np.array(np.kron(img, repeat), dtype=np.uint8)
    cv2.imwrite("NNinterp.png", img)
    return img

def BLinterpol(img, M, N):
    img = cv2.imread("../imgs/cameraman.tif", cv2.IMREAD_GRAYSCALE)
    R = img.shape[0]
    C = img.shape[1]
    # Scale factor
    k = M/R
    out = np.zeros((M, N))
    idx_cols = np.zeros(N)
    idx_rows = np.zeros(M)
    idx_cols[0:N:k] = 1
    idx_rows[0:M:k] = 1
    idx_out = np.outer(idx_cols, idx_rows)
    not_idx_out = np.logical_not(idx_out)
    # Created skeletal frame
    NNint = NNinterpol(img, M, N)
    col_shifted_NNint = np.roll(NNint,N-k,axis=1)
    col_shifted_NNint[:,N-k:] = 0
    row_shifted_NNint = np.roll(NNint, M-k, axis=0)
    row_shifted_NNint[M-k:,:] = 0
    tmp = np.multiply(NNint, idx_out)

    weights_row_inter = np.kron(np.ones(C),np.arange(0,1,1.0/k))
    rev_idx_row_inter = weights_row_inter[::-1]
    weights_col_inter = np.kron(np.ones(R),np.arange(0,1,1.0/k))
    
    # Interpolating the columns
    out = weights_row_inter * col_shifted_NNint + (1-weights_row_inter) * NNint
    col_mask = np.kron(idx_cols, np.ones(N)).reshape((1024, 1024))
    out_col = out * col_mask
    selected_rows = out_col[0:M:k,:]
    selected_rows = np.repeat(selected_rows, k, axis = 0)
    shifted_selected_rows = np.roll(selected_rows, M-k, axis=0)
    shifted_selected_rows[M-k:,:] = 0

    out = (weights_col_inter * shifted_selected_rows.T + (1-weights_col_inter) * selected_rows.T).T
    out = np.array(out, dtype = np.uint8)

    cv2.imwrite("BLinterp.png", out)
    #cv2.imshow("image", out)
    #cv2.waitKey()

def cart2pol(img, size):
    M = size[0]
    N = size[1]
    rows = np.arange(M)
    cols = np.arange(N)
    r_idx = np.sqrt((rows - M/2)*(rows-M/2) + (cols-N/2)*(cols-N/2)) * 2*D/M
    theta_idx = (np.arctan2(cols - N/2, rows - N/2) / math.pi + 0.5) * Theta

if __name__ == '__main__':
    img = cv2.imread("../imgs/cameraman.tif", cv2.IMREAD_GRAYSCALE)
    BLinterpol(img, 1024, 1024)
    #NNinterpol(img, 1024, 1024)
