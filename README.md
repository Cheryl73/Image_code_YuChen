# Image Processor

# Overview

This Python script is designed for processing and analyzing microscopy images stored in .czi files. It provides functionalities for image slicing, intensity analysis, thickness measurement, and overall image generation. The script uses the aicsimageio library to read CZI files, OpenCV for image processing, and Matplotlib for visualization.
Dependencies

Make sure you have the following Python libraries installed:

    matplotlib
    aicsimageio
    cv2 (OpenCV)
    numpy
    glob
    tqdm
    pickle
    tkinter

You can install them using:

bash

pip install matplotlib aicsimageio opencv-python numpy glob2 tqdm

Usage
Basic Usage

python

from image_processor import ImageProcessor

# Example usage
image_processor = ImageProcessor()

# Save image slices
image_processor.save_figs(im_path="path/to/your/image.czi", savepath="path/to/save/slices/", slice_dir="Z")

# Get overall mean intensity
mean_intensity = image_processor.get_overall_mean("path/to/your/image.czi", channel=0)
print("Overall Mean Intensity:", mean_intensity)

Advanced Usage

python

from image_processor import ImageProcessor

# Example advanced usage
image_processor = ImageProcessor()

# Slice and save images along the X direction
image_processor.save_figs(im_path="path/to/your/image.czi", savepath="path/to/save/slices/", slice_dir="X", slice_idx=[0, 1, 2])

# Count cells in a specific channel and save the result
cell_info = image_processor.count_cell("path/to/your/image.czi", channel=0, save_path="path/to/save/cell_map.tiff")
print("Cell Information:", cell_info)

Documentation
ImageProcessor Class
Methods

    save_figs(im_path, savepath, slice_dir="Z", slice_idx=[], brightness_factor=[2, 2, 2]): Saves image slices along a specified direction.

    showImage(title, img, ctype): Displays images using Matplotlib.

    get_coverage(im): Calculates the coverage of a binary image.

    get_thickness(czi_path, method="count", chan=0): Measures the thickness of structures in a CZI file.

    get_overall_Z(czi_path, channel=0, save_path=None): Generates an overall image by summing slices along the Z direction.

    count_cell(czi_path, channel=0, save_path=None): Counts cells in a CZI file using image segmentation.

    binarize_Z(czi_path, chan=0): Binarizes slices along the Z direction.

    get_overall_mean(czi_path, channel=0): Calculates the overall mean intensity of a specified channel.

    statistic_ans(czi_path, overlap="layer", chan=0): Analyzes image statistics, including area, thickness, and intensity.

    treat_raw_data(data_path, savepath, overlap="layer", chan=0): Processes raw CZI data and saves the results in a pickle file.

Example Usage

python

from image_processor import ImageProcessor

# Example usage
image_processor = ImageProcessor()

# Save image slices along the Z direction
image_processor.save_figs(im_path="path/to/your/image.czi", savepath="path/to/save/slices/", slice_dir="Z")

# Calculate overall mean intensity of channel 0
mean_intensity = image_processor.get_overall_mean("path/to/your/image.czi", channel=0)
print("Overall Mean Intensity:", mean_intensity)

# Count cells in channel 1 and save the result
cell_info = image_processor.count_cell("path/to/your/image.czi", channel=1, save_path="path/to/save/cell_map.tiff")
print("Cell Information:", cell_info)

Notes

    Ensure that your CZI files are in the correct format and accessible.
    Adjust parameters such as brightness_factor and slice_idx based on your specific requirements.

Feel free to customize and integrate these functions into your image processing pipeline.
