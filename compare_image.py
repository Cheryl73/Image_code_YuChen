from image_pao import ImageProcessor
import run
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file1 = run.getfile("select image file 1:")
    file2 = run.getfile("select image file 2:")

    im1 = ImageProcessor.get_overall_Z_max(file1, 0)
    im2 = ImageProcessor.get_overall_Z_max(file2, 0)
    cv.imshow("image 1", im1)
    cv.imshow("image 2", im2)
