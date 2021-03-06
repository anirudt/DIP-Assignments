import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb
from plotter import graphify

def graphify(x, y, xlabel, ylabel, title, name):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.stem(x, y)
    plt.grid = True
    plt.savefig(name+'.png')
    plt.show()

def histogrammer(img):
  hist = np.zeros(256)

  # Computes the histogram.
  col = 0
  for num in range(img.shape[0]):
      hist[img[num]] += 1
  print hist

  graphify(range(hist.shape[0]), hist, "", "Histogram","Histogram", "hist")

def proc():
  # Load the image
  img = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE).flatten()

  # Initialize the histogram
  histogrammer(img)

if __name__ == '__main__':
    proc()
