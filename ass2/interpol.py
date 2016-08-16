#! /usr/bin/python
import cProfile
import cv2
import numpy as np
import pdb
import optparse


parser = optparse.OptionParser()
parser.add_option("-a", action="store_true", default = False, dest="read_czp")
parser.add_option("-b", action="store_true", default = False, dest="nnint")
parser.add_option("-c", action="store_true", default = False, dest="blint")
(opts, args) = parser.parse_args()

def read_czp():
    f = open("test_polar.czp", "rb")
    header = f.read(16)
    status_flag = int((f.read(1)).encode("hex"),16)

    byte = f.read(8)[::-1]
    R = int(byte.encode('hex'), 16)
    byte = f.read(8)[::-1]
    T = int(byte.encode("hex"), 16)

    bpp = int(f.read(1).encode("hex"), 16)
    
    byte = "."
    img = np.zeros((R, T), np.uint8)
    for r in xrange(R):
        for t in xrange(T):
            byte = f.read(2)
            val = int(byte.encode("hex"), 16)
            img[r, t] = val * np.iinfo(np.uint8).max / np.iinfo(np.uint16).max

    M = N = R*2
    cart_img = pol2cart(img, [M, N], 0)
    cv2.imwrite("test_cart_wo_vec.bmp", cart_img)

    cart_img = pol2cart(img, [M, N], 1)
    cv2.imwrite("test_cart_vec.bmp", cart_img)


def NNinterpol(img, size):
    M = size[0]
    N = size[1]
    R = img.shape[0]
    C = img.shape[1]
    j = np.floor(M/R)
    k = np.floor(N/C)
    repeat = np.ones((j, k))
    img = np.array(np.kron(img, repeat), dtype=np.uint8)
    cv2.imwrite("NNinterp.png", img)
    return img

def BLinterpol(img, size):
    M = size[0]
    N = size[1]
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
    NNint = NNinterpol(img,[M, N])
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

def cart2pol(img, theta_points, vec):
    M = img.shape[0]; N = img.shape[1];
    if not vec:
        T = theta_points; R = min(M/2, N/2)
        pol_img = np.zeros((M/2, T))
        for row in xrange(M):
            for col in xrange(N):
                r = int(np.sqrt((row-M/2)**2 + (col-N/2)**2))
                t = int((np.arctan2(col-N/2, row-M/2)+np.pi)*T/(2*np.pi))
                if r >= R or t >= T:
                    continue
                else:
                    pol_img[r,t] = img[row, col]
        return pol_img
    
    else:
        theta, R = np.meshgrid(np.linspace(0, 2*np.pi, theta_points),
                               np.arange(min(M/2, N/2)))
        cart_x = R * np.cos(theta) + M/2
        cart_y = R * np.cos(theta) + N/2
        cart_x = cart_x.astype(int)
        cart_y = cart_y.astype(int)

        polar_img = img[cart_x, cart_y]

        # Arrange the image data in R-Theta form.
        polar_img = polar_img.reshape((min(M/2, N/2), theta_points))
        return polar_img



def pol2cart(img, size, vec):
    M = size[0]; N = size[1];
    R = img.shape[0]; T = img.shape[1];
    if not vec:
        cart_img = np.zeros((M, N), dtype=np.uint8)
        for r in xrange(R):
            for t in xrange(T):
                x = int(r*np.cos(t*2*np.pi/T)+M/2)
                y = int(r*np.sin(t*2*np.pi/T)+N/2)
                cart_img[x, y] = img[r, t]
        return cart_img

    else:
        X, Y = np.meshgrid(np.arange(M), np.arange(N))
        r = np.sqrt((X-M/2)**2 + (Y-N/2)**2)
        theta = (np.arctan2(Y-N/2, X-M/2)+np.pi)*T/(2*np.pi)
        r = r.astype(int)
        theta = theta.astype(int)
        print img.shape
        print r.shape
        r_clip = np.clip(r, 0, R-1)
        r_mask = np.array(r <= r_clip, dtype=np.uint8)
        theta = theta % T
        cart_img = img[r_clip, theta]
        cart_img = np.multiply(cart_img, r_mask)
        cart_img.reshape((M, N))
        return cart_img


if __name__ == '__main__':
    if opts.read_czp:
        read_czp()
    if opts.nnint:
        img = cv2.imread("../imgs/cameraman.tif", cv2.IMREAD_GRAYSCALE)
        NNinterpol(img, [1024, 1024])
    if opts.blint:
        img = cv2.imread("../imgs/cameraman.tif", cv2.IMREAD_GRAYSCALE)
        BLinterpol(img, [1024, 1024])
    #BLinterpol(img, [1024, 1024])
    #pol_img = cart2pol(img, [256, 256], 0, 128, 2000)
    #pol2cart()
