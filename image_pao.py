import matplotlib.image
from aicsimageio import AICSImage
import cv2 as cv
import numpy as np
import glob
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from npTec import np2tec
import tkinter as tk
import csv
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
import os
from collections import defaultdict
from skimage.segmentation import watershed as skwater
from data_tree import data_tree


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def save_figs(im_path, savepath, slice_dir="Z", slice_idx=[], brightness_factor=[2, 2, 2]):
        '''

        :param im_path: .czi file path
        :param savepath: result saving folder
        :param slice_dir: which (X, Y, Z) direction to cut
        :param slice_idx: index to cut
        :param brightness_factor: in order of [G,B,R]
        :return: None
        '''
        name = ImageProcessor.get_filename(im_path)
        img = AICSImage(im_path)
        dim = img.dims
        i = 0
        if slice_dir == "Z":
            if slice_idx:
                items = slice_idx
            else:
                items = range(0, dim.Z)
            for i in items:
                im_0 = img.get_image_data("YX", C=0, T=0, Z=i) * brightness_factor[0]
                try:
                    im_1 = img.get_image_data("YX", C=1, T=0, Z=i) * brightness_factor[1]
                except:
                    print("No channel 1")
                    im_1 = im_0 * 0
                try:
                    im_2 = img.get_image_data("YX", C=2, T=0, Z=i) * brightness_factor[2]
                except:
                    print("No channel 2")
                    im_2 = im_0 * 0
                merged = cv.merge([im_0, im_1, im_2])
                # merged = merged * brightness_factor
                # cv.imshow("Overall", merged)
                filename = name + "_Z_" + str(i) + ".tiff"
                path = os.path.join(savepath, filename)
                cv.imwrite(path, merged)
        elif slice_dir == "X":
            if slice_idx:
                items = slice_idx
            else:
                items = range(0, dim.X)
            for i in items:
                im_0 = img.get_image_data("ZY", C=0, T=0, X=i) * brightness_factor[0]
                try:
                    im_1 = img.get_image_data("ZY", C=1, T=0, X=i) * brightness_factor[1]
                except:
                    print("No channel 1")
                    im_1 = im_0 * 0
                try:
                    im_2 = img.get_image_data("ZY", C=2, T=0, X=i) * brightness_factor[2]
                except:
                    print("No channel 2")
                    im_2 = im_0 * 0
                merged = cv.merge([im_0, im_1, im_2])
                # merged = merged * brightness_factor
                filename = name + "_X_" + str(i) + ".tiff"
                path = os.path.join(savepath, filename)
                cv.imwrite(path, merged)
        elif slice_dir == "Y":
            if slice_idx:
                items = slice_idx
            else:
                items = range(0, dim.Y)
            for i in items:
                im_0 = img.get_image_data("ZX", C=0, T=0, Y=i) * brightness_factor[0]
                try:
                    im_1 = img.get_image_data("ZX", C=1, T=0, Y=i) * brightness_factor[1]
                except:
                    print("No channel 1")
                    im_1 = im_0 * 0
                try:
                    im_2 = img.get_image_data("ZX", C=2, T=0, Y=i) * brightness_factor[2]
                except:
                    print("No channel 2")
                    im_2 = im_0 * 0
                merged = cv.merge([im_0, im_1, im_2])
                # merged = merged * brightness_factor
                filename = name + "_Y_" + str(i) + ".tiff"
                path = os.path.join(savepath, filename)
                cv.imwrite(path, merged)
        else:
            print("Check dimension name, nothing saved!")
            return

        print(str(len(items)) + " images saved!")
        return

    @staticmethod
    def showImage(title, img, ctype):
        plt.figure(figsize=(10, 10))
        if ctype == "bgr":
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            plt.imshow(rgb)
        elif ctype == "hsv":
            hsv = cv.cvtColor(img, cv.COLOR_HSV2RGB)
            plt.imshow(hsv)
        elif ctype == "gray":
            plt.imshow(img, cmap="gray")
        elif ctype == "rgb":
            plt.imshow(img)
        else:
            raise Exception("Unknown color type.")
        plt.title(title)
        plt.show()

    @staticmethod
    def get_coverage(im):
        threshold, im2 = cv.threshold(im, 0, 1, cv.THRESH_OTSU)
        area = np.sum(im2)
        return area

    @staticmethod
    def get_thickness(czi_path, method="count", chan=0, thresh_min=2000):
        img = AICSImage(czi_path)
        dim = img.dims
        thickness = np.zeros([dim.Y, dim.X])
        im2, _ = ImageProcessor.binarize_Z(czi_path, chan=chan, th_min=thresh_min)
        if method == "count":
            thickness_ = np.sum(im2, axis=0)
        else:
            for x in tqdm(range(0, dim.X)):
                for y in range(0, dim.Y):
                    z_colum = im2[:, x, y]
                    space = ImageProcessor.get_range(z_colum)
                    thickness[x, y] = space[2]
        return thickness_

    @staticmethod
    def get_dim(czi_path):
        img = AICSImage(czi_path)
        dim = img.dims
        return [dim.X, dim.Y, dim.Z]

    @staticmethod
    def get_range(arr):
        edge_l = -1
        edge_r = -1
        for i, n in enumerate(arr):
            if n > 0:
                if edge_l < 0:
                    edge_l = i
                edge_r = i
        if edge_l < 0:
            rst = (0, 0, 0)
        else:
            rst = (edge_l, edge_r, edge_r - edge_l + 1)
        return rst

    @staticmethod
    def get_overall_Z(czi_path, channel=0, save_path=None):
        img = AICSImage(czi_path)
        dim = img.dims
        im_all = 0
        for i in range(0, dim.Z):
            im = img.get_image_data("YX", C=channel, T=0, Z=i)
            im_all = cv.add(im_all, im)
        if save_path:
            matplotlib.image.imsave(save_path, im_all)
        return im_all

    @staticmethod
    def get_overall_Z_max(czi_path, chan=0, save_path=None):
        img = AICSImage(czi_path)
        dim = img.dims
        im_data = img.data[0, chan, :]
        im_all = im_data.max(axis=0)
        if save_path:
            matplotlib.image.imsave(save_path, im_all)
        return im_all

    @staticmethod
    def get_data(czi_path):
        img = AICSImage(czi_path)
        return img.data

    @staticmethod
    def count_cell(czi_path, channel=0, save_path=None):
        im_all = ImageProcessor.get_overall_Z_max(czi_path, channel)
        # ImageProcessor.showImage("Original", im_all, "rgb")
        # im_all = cv.cvtColor(im_all, cv.COLOR_BGR2GRAY)
        th, thresh = cv.threshold(im_all, 0, 255, cv.THRESH_OTSU)
        # ImageProcessor.showImage("Grayscale", gray, "gray")
        # ImageProcessor.showImage("Applying Otsu", thresh, "gray")
        median = cv.medianBlur(thresh, 5)
        # ImageProcessor.showImage('MedianBlur', median, 'gray')
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv.dilate(median, kernel, iterations=8)
        # ImageProcessor.showImage('Dilated', dilated, 'gray')
        dilated = np.uint8(dilated)
        dist = cv.distanceTransform(dilated, cv.DIST_L2, 5)
        # ImageProcessor.showImage('Distance', dist, 'gray')
        fraction_foreground = 0.5
        ret, sure_fg = cv.threshold(dist, fraction_foreground * dist.max(), 255, 0)
        # ImageProcessor.showImage('Surely Foreground', sure_fg, 'gray')
        unknown = cv.subtract(dilated, sure_fg.astype(np.uint8))
        # ImageProcessor.showImage('Unknown', unknown, 'gray')

        ret, markers = cv.connectedComponents(sure_fg.astype(np.uint8))
        # ImageProcessor.showImage('Connected Components', markers, 'rgb')

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == np.max(unknown)] = 0

        # ImageProcessor.showImage('markers', markers, 'rgb')

        dist = cv.distanceTransform(dilated, cv.DIST_L2, 5)
        markers = skwater(-dist, markers, watershed_line=True)


        rst = []
        for i in range(2, np.max(markers) + 1):
            idx = np.argwhere(markers == i)
            area = len(idx)
            y_min = np.min(idx[:, 0])
            y_max = np.max(idx[:, 0])
            x_min = np.min(idx[:, 1])
            x_max = np.max(idx[:, 1])
            rg = [x_min, x_max, y_min, y_max]
            rst.append([area, rg])

        # # text cell number on each cell center
        # markers_labeled = markers.copy()
        # for i in range(2, np.max(markers) + 1):
        #     idx = np.argwhere(markers == i)
        #     y = int(np.mean(idx[:, 0]))
        #     x = int(np.mean(idx[:, 1]))
        #     cv.putText(markers_labeled, str(i - 1), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ImageProcessor.showImage('Watershed', markers, 'rgb')
        # print(len(set(markers.flatten()))-1)
        # print(np.max(markers) - 1)

        if save_path:
            matplotlib.image.imsave(save_path, markers)
            # matplotlib.image.imsave(save_path.replace(".tiff", "_labeled.tiff"), markers_labeled)

        return rst, markers

    @staticmethod
    def divide_cell(czi_path, target_chan, cell_chan, thresh_min, save_path=None):
        positions, cell_map = ImageProcessor.count_cell(czi_path, channel=cell_chan)
        target_3d, _ = ImageProcessor.binarize_Z(czi_path, chan=target_chan, th_min=thresh_min)
        cell_3d, _ = ImageProcessor.binarize_Z(czi_path, chan=cell_chan, th_min=thresh_min)
        rst = []
        for i in range(2, np.max(cell_map) + 1):
            mask = np.zeros(cell_3d.shape)
            idx = np.argwhere(cell_map == i)
            mask[:, idx[:, 0], idx[:, 1]] = 1
            cell_zone = cell_3d * mask
            target_zone = target_3d * mask
            cell_2D = np.max(cell_zone, axis=2)
            idx = np.argwhere(cell_2D == 1)
            z_min = np.min(idx[:, 0])
            z_max = np.max(idx[:, 0])
            mid_idx = round(0.51 * (z_min + z_max))
            # target_2D = np.max(target_zone, axis=2)
            up_side = target_zone[0:mid_idx, :, :]
            down_side = target_zone[mid_idx:, :, :]
            # area = np.sum(mask[0,:,:])
            area = positions[i - 2][0]
            volume_up = np.sum(up_side)
            volume_down = np.sum(down_side)
            thickness_up = volume_up / area
            thickness_down = volume_down / area
            cur = [volume_up, volume_down, thickness_up, thickness_down, area]
            rst.append(cur)
            # ImageProcessor.showImage("cell 2D", cell_2D, ctype='gray')
            # ImageProcessor.showImage("target 2D", target_2D, ctype='gray')
            # iii = np.argwhere(mask == 1)
            # position = positions[i - 2]

        # Now consider no-cell area
        mask = np.zeros(cell_3d.shape)
        idx = np.argwhere((cell_map == 0) | (cell_map == 1))
        mask[:, idx[:, 0], idx[:, 1]] = 1
        target_zone = target_3d * mask
        area = np.sum(mask[0, :, :])
        volume = np.sum(target_zone)
        thickness_single = volume / area
        single = [volume, thickness_single, area]
        rst_array = np.array(rst)
        cell_zone_ave = np.average(rst_array, axis=0)
        # ImageProcessor.showImage("mask", mask[0,:,:], ctype='gray')
        number_of_cells = len(rst)

        return cell_zone_ave, single, number_of_cells

    @staticmethod
    def get_filename(czi_path):
        name = czi_path.split("\\")[-1]
        name = name.split(".")[0]
        return name

    @staticmethod
    def binarize_Z(czi_path, chan=0, th_min=2000, savepath=None):
        img = AICSImage(czi_path)
        dim = img.dims
        rst = np.zeros([dim.Z, dim.Y, dim.X])
        ths = []
        for z in range(0, dim.Z):
            im = img.get_image_data("YX", C=chan, T=0, Z=z)
            # img_scaled = cv.normalize(im, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)
            # cv.imshow('im_scaled', img_scaled)
            # im_8 = im.astype(dtype='uint8')
            # dst = cv.equalizeHist(im_8)
            # cv.imshow('aaa', im_8)
            # cv.imshow('bbb', dst)
            th, im2 = cv.threshold(im, 0, 1, cv.THRESH_OTSU)
            ths.append(th)
            if th < th_min:
                th, im2 = cv.threshold(im, th_min, 1, cv.THRESH_BINARY)
            rst[z, :, :] = im2
            # th_2, im2_2 = cv.threshold(dst, 0, 1, cv.THRESH_OTSU)
            # plt.figure()
            # plt.imshow(im2, cmap='gray')
            # plt.figure()
            # plt.imshow(im2_2, cmap='gray')
            # plt.show()
            # area1 = np.sum(im2)
            # area2 = np.sum(im2_2)
            # print("origin:", area1, "     equalize:", area2)
            # print("threshold:", th)

        if savepath:
            plt.figure()
            plt.plot(ths, marker='o')
            plt.axhline(y=th_min, color='r', linestyle='-')
            plt.xlabel("z")
            plt.ylabel("Threshold")
            plt.savefig(savepath)
            plt.close()

        return rst, ths

    @staticmethod
    def get_overall_mean(czi_path, channel=0):
        img = AICSImage(czi_path)
        dim = img.dims
        im_data = img.data[0, channel, :]
        rst = np.average(im_data)
        return rst

    @staticmethod
    def get_overall_image(path, savepath=None):
        try:
            im_ch0 = ImageProcessor.get_overall_Z_max(path, 0)
        except:
            print("No image data loaded, check data dimension!")
            return [], [], [], []
        try:
            im_ch1 = ImageProcessor.get_overall_Z_max(path, 1)
        except:
            im_ch1 = 0 * im_ch0
        try:
            im_ch2 = ImageProcessor.get_overall_Z_max(path, 2)
        except:
            im_ch2 = 0 * im_ch0
        # cv.imshow("Green", im_G)
        # cv.imshow("Blue", im_B)
        # cv.imshow("Red", im_R)
        im_0 = 0 * im_ch0
        im_ch0_0 = cv.merge([im_ch0, im_0, im_0])
        im_ch1_0 = cv.merge([im_0, im_ch1, im_0])
        im_ch2_0 = cv.merge([im_0, im_0, im_ch2])
        im_all = cv.merge([im_ch0, im_ch1, im_ch2])

        if savepath:
            path_all = savepath + "_all.tiff"
            path_0 = savepath + "_chan0.tiff"
            path_1 = savepath + "_chan1.tiff"
            path_2 = savepath + "_chan2.tiff"
            cv.imwrite(path_all, im_all)
            cv.imwrite(path_0, im_ch0_0)
            cv.imwrite(path_1, im_ch1_0)
            cv.imwrite(path_2, im_ch2_0)

        return im_all, im_ch0, im_ch1, im_ch2

    @staticmethod
    def statistic_ans(czi_path, overlap="layer", chan=0, thresh_min=2000):
        '''
        :param overlap: overlap method: maximum or binarize by layer
        :param czi_path: file path of .czi image
        :param chan: channel number to analysis
        :param thresh_min: minimum threshold when binarize the image layer
        :return:
        '''
        # print("Now treat: ", czi_path)
        name = ImageProcessor.get_filename(czi_path)
        # img = AICSImage(czi_path)
        # dim = img.dims
        if overlap == "max":
            im_all = ImageProcessor.get_overall_Z_max(czi_path, chan=chan)
            # cv.imshow("lalala", im_all)
            th, im2 = cv.threshold(im_all, 0, 1, cv.THRESH_OTSU)
            print("threshold for all image: ", th)
            area = np.count_nonzero(im2)
            bi_z = None
        else:
            path = ''.join(czi_path.split("\\")[0:-1]) + "/result/binarize_images/" + "chan_" + str(
                chan) + '_' + name + "_threshold.tiff"
            bi_z, _ = ImageProcessor.binarize_Z(czi_path, chan=chan, th_min=thresh_min, savepath=path)
            im2 = bi_z.max(axis=0)
            area = np.count_nonzero(im2)

        path = ''.join(czi_path.split("\\")[0:-1]) + "/result/binarize_images/" + "chan_" + str(
            chan) + '_' + name + ".tiff"
        matplotlib.image.imsave(path, im2)

        # bi_z = ImageProcessor.binarize_Z(czi_path)
        # im2_2 = bi_z.max(axis=0)
        # area = np.count_nonzero(bi_z)
        #
        # cv.imshow("im2_max", im2*255*255)
        # cv.imshow("im2_layers", im2_2*255*255)
        # cv.imshow("im_all", im_all)

        thickness = ImageProcessor.get_thickness(czi_path, method="count", chan=chan, thresh_min=thresh_min)
        thickness_mask = ImageProcessor.mask_thickness_by_area(thickness, im2)
        thickness_ave = np.sum(thickness_mask) / area

        im_all = ImageProcessor.get_overall_Z_max(czi_path, chan=chan)
        intensity = np.average(im_all)
        # intensity = ImageProcessor.get_overall_mean(czi_path, channel=chan)

        # cv.imshow("t1", thickness)
        # cv.imshow("t2", thickness_mask)

        print("area:", area)
        print("thickness:", thickness_ave)
        print("average intensity:", intensity)
        return area, thickness_mask, intensity, bi_z

    @staticmethod
    def mask_thickness_by_area(thickness, area_bi):
        return thickness * area_bi

    @staticmethod
    def treat_raw_data(data_directory, filepath, nc_files=None, overlap="layer", chan=1, cell_chan=0, thresh_min=2000):
        if nc_files is None:
            nc_files = []
            is_nc = False
        else:
            path = ''.join(data_directory + "/result/negative_control/" + "average_NC.tiff")
            print("Calculating average image of negative control...")
            im_nc = ImageProcessor.get_sample_average(nc_files, path)
            im_nc = im_nc[:, :, chan]
            is_nc = True
        files = glob.glob(data_directory + "/*.czi")
        rst = {}
        nc_f = set(nc_files)
        for i, f in enumerate(files):
            try:
                print("(", str((i + 1)) + '/' + str(len(files)), ")", "Now treat: ", f)
                f_ = f.replace('\\', '/')
                if f_ in nc_f:
                    print("Warning: ", f, "This file is chosen as negative control file.")
                    # continue
                if is_nc:
                    print("calculating minimum threshold...")
                    im_max = ImageProcessor.get_overall_Z_max(f, chan=chan)
                    name = ImageProcessor.get_filename(f)
                    savepath = data_directory + "/result/negative_control/" + "chan_" + str(
                        chan) + '_' + name + "_threshold.tiff"
                    thresh_min = ImageProcessor.get_imageIntersection(im_nc, im_max, savepath)
                print("minimum threshold: ", thresh_min)
                a, h, I, bi_z = ImageProcessor.statistic_ans(f, overlap=overlap, chan=chan, thresh_min=thresh_min)
                cell_zone, single, number_of_cells = ImageProcessor.divide_cell(f, target_chan=chan,
                                                               cell_chan=cell_chan, thresh_min=thresh_min)
                dims = ImageProcessor.get_dim(f)
                name = ImageProcessor.get_filename(f)
                thickness_up = cell_zone[2]
                thickness_down = cell_zone[3]
                thickness_single = single[1]
                rst[name] = (a, h, I, thickness_single, thickness_down, thickness_up, dims, thresh_min, chan, cell_chan, number_of_cells)
                print("Number of cells: ", number_of_cells)
            except:
                print(f, "is not valid")
                errorfile = data_directory + "/error.log"
                with open(errorfile, "a") as file:
                    file.write(f)
                    file.write("\n")
        with open(filepath, 'wb') as handle:
            pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data is treated and save as " + filepath)

    @staticmethod
    def get_cross(czi_path, chans):
        rst = 1
        for chan in chans:
            bi_z, _ = ImageProcessor.binarize_Z(czi_path, chan=chan)
            v = np.count_nonzero(bi_z)
            print("channel ", chan, " Volume:", v)
            rst = rst * bi_z
        s = np.sum(rst)
        print("Number of overlapping pixels: ", s)
        return rst

    @staticmethod
    def get_cross_with_min_threshold(czi_path, chans, th_mins):
        rst = 1
        for i, chan in enumerate(chans):
            th_min = th_mins[chan]
            bi_z, _ = ImageProcessor.binarize_Z(czi_path, chan=chan, th_min=th_min)
            v = np.count_nonzero(bi_z)
            print("channel ", chan, " Volume:", v)
            rst = rst * bi_z
        s = np.sum(rst)
        print("Number of overlapping pixels: ", s)
        return rst

    @staticmethod
    def get_cross_from_rst(result_files, save_path):
        rsts = {}
        rst_names = None
        valid_names = set()
        for file in result_files:
            with open(file, 'rb') as handle:
                rst = pickle.load(handle)
            names = list(rst.keys())
            if not names:
                print(file, ":No result found, check result file!")
                continue
            if not rst_names:
                rst_names = set(names)
                valid_names = rst_names
            else:
                names = set(names)
                valid_names = names & rst_names
                diff_names = names ^ rst_names
                if diff_names:
                    for nm in diff_names:
                        print(nm, "is not in all result file, will be ignored!")
            err_names = []
            for i, name in enumerate(valid_names):
                s = "(" + str(i) + "/" + str(len(valid_names)) + ")" + " Now treat:"
                print(s, name)
                rst_cur = rsts.get(name, 1)
                bi_z = rst[name][7]
                try:
                    rst_cur = rst_cur * bi_z
                except:
                    print(name, "result file not in same dimension!")
                    err_names.append(name)
                    continue
                rsts[name] = rst_cur
        file_headers = ["Name", "Area", "Thickness"]
        filepath = save_path
        err_names = set(err_names)
        with open(filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(file_headers)
            for name in valid_names:
                if name in err_names:
                    continue
                dt = rsts[name]
                im2 = dt.max(axis=0)
                area = np.count_nonzero(im2)
                volume = np.sum(dt)
                thickness = volume / area
                writer.writerow([name, area, thickness])

        return

    @staticmethod
    def get_hist_percentage(x, hist, percentage):
        hist_norm = hist / np.sum(hist)
        cur = 0
        i = 0
        while cur < percentage and i < len(hist):
            cur += hist_norm[i]
            i += 1
        idx = min(i, len(hist))
        return idx

    @staticmethod
    def get_imageIntersection(im1, im2, savepath=None):
        hist1, bins1 = np.histogram(im1.ravel(), bins=200)
        hist2, bins2 = np.histogram(im2.ravel(), bins=bins1)
        x = bins1[1:]
        y1 = hist1
        y2 = hist2
        idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
        idx = list(idx.ravel())
        idx_95 = ImageProcessor.get_hist_percentage(x, y1, 0.95)
        threshold_95 = x[idx_95]
        if not idx:
            threshold = threshold_95
            threshold_idx = idx_95
        else:
            cur = float('inf')
            cur_i = -1
            for i in idx:
                if abs(x[i] - threshold_95) < cur:
                    cur = abs(x[i] - threshold_95)
                    cur_i = i
            threshold = x[cur_i]
            threshold_idx = cur_i
        if savepath:
            plt.figure()
            plt.plot(x, y1, label="Negative control")
            plt.plot(x, y2, label="Sample")
            plt.plot(x[idx], y2[idx], 'ro', ms=7, label='intersection')
            plt.plot(x[threshold_idx], y2[threshold_idx], 'bo', ms=7, label='threshold')
            plt.legend()
            plt.savefig(savepath)
            plt.close()
        return threshold

    @staticmethod
    def get_sample_average(samples, savepath=None):
        images = []
        for sample in samples:
            image, _, _, _ = ImageProcessor.get_overall_image(sample)
            if len(image) == 0:
                print(sample, "not valid!")
                continue
            images.append(image)
        if not images:
            print("No negative control image found!")
            return
        s = np.zeros(images[0].shape)
        for i in range(0, len(images)):
            s = s + images[i] / len(images)
        s = s.astype(images[0].dtype)
        if savepath:
            cv.imwrite(savepath, s)
        return s


def treat_raw_data(data_path, savepath, nc_files=None, overlap="layer", chan=0, cell_chan=0, thresh_min=2000):
    ImageProcessor.treat_raw_data(data_path, savepath, nc_files=nc_files, overlap=overlap,
                                  chan=chan, cell_chan=cell_chan, thresh_min=thresh_min)


def get_name_info(names):
    def treat_name(nm):
        symbl = []
        value = []
        cur = ''
        for s in nm:
            if s.isdigit():
                cur += s
            else:
                symbl.append(s)
                if cur:
                    value.append(int(cur))
                cur = ''
        value.append(int(cur))
        return symbl, value

    info = defaultdict(set)
    smb = []
    for name in names:
        name = name.split("_")[-1]
        smb, value = treat_name(name)
        for i, sb in enumerate(smb):
            info[sb].add(value[i])

    for key in info:
        info[key] = list(info[key])
        info[key].sort()
    return info, smb


def get_file_key(info: list[int, int, int], order, pre=None):
    '''
    Generate key from data information to get result from result file
    :param pre: prefix
    :param info: [concentrate, sample #, image #]
    :param order: i.e. ["C", "S", "I"], name if each section
    :return: key in the result pickle
    '''
    code = order[0] + "{:0>3d}".format(info[0]) + order[1] + "{:d}".format(info[1]) + order[2] + "{:d}".format(info[2])
    if pre:
        code = pre + "_" + code
    return code


def group_analyse2(filepath):
    with open(filepath, 'rb') as handle:
        rst = pickle.load(handle)
    names = rst.keys()
    if not names:
        print("No result found, check result file!")
        return
    dt_h = data_tree()
    dt_area = data_tree()
    dt_v = data_tree()
    dt_i = data_tree()
    for name in names:
        dt = rst[name]
        area = dt[0]
        h = np.sum(dt[1]) / area
        v = area * h
        intensity = dt[2]
        dt_h.add_data(name, h)
        dt_v.add_data(name, v)
        dt_area.add_data(name, area)
        dt_i.add_data(name, intensity)
    print("================ area ==================")
    dt_area.print_data_recursive()
    print("================ Volume ================")
    dt_v.print_data_recursive()
    print("=============== thickness ==============")
    dt_h.print_data_recursive()
    print("=============== Intensity ==============")
    dt_i.print_data_recursive()


def read_rst(rst_path):
    with open(rst_path, 'rb') as handle:
        rst = pickle.load(handle)
    names = rst.keys()
    if not names:
        print("No result found, check result file!")
        return
    file_headers = ["Name", "Intensity", "Area", "Thickness_overall", "Thickness_single", "Thickness_down",
                    "Thickness_up", "Dimension", "Area percentage", "Number of cells"]
    filepath = os.path.splitext(rst_path)[0] + ".csv"
    names = rst.keys()
    n = len(names)
    print("Reading from: " + rst_path)
    print(str(n) + " items extracted...")
    with open(filepath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(file_headers)
        for name in names:
            dt = rst[name]
            area = dt[0]
            h = np.sum(dt[1]) / area
            v = area * h
            intensity = dt[2]
            thickness_single = dt[3]
            thickness_down = dt[4]
            thickness_up = dt[5]
            dims = dt[6]
            area_percent = area / (dims[0] * dims[1])
            number_of_cells = dt[10]
            writer.writerow([name, intensity, area, h, thickness_single,
                             thickness_down, thickness_up, dims, area_percent, number_of_cells])
    print("Result exported as " + filepath)


def get_thickness_image(rootpath, datapath):
    with open(datapath, 'rb') as handle:
        rst = pickle.load(handle)
    savepath = os.path.join(rootpath, "result/thickness_map/")
    os.makedirs(savepath, exist_ok=True)
    for i, f in tqdm(enumerate(rst)):
        result = rst[f]
        thick = result[1]
        thick_rot = np.rot90(thick, -1)
        # cv.imshow("lala", thick)
        size = np.shape(thick)
        x = np.arange(0, size[0], 1)
        y = np.arange(0, size[1], 1)
        z = [0]
        X, Y = np.meshgrid(x, y)

        var = [x, y, z]
        data = [np.reshape(thick_rot, (len(x), len(y), 1))]
        names = [["x", "y", "z"], ["Thickness"]]
        name = ImageProcessor.get_filename(f)
        filepath = os.path.join(savepath, name + ".dat")
        np2tec(var, data, names, filepath)

        vmin = 0
        vmax = 30
        fig, ax = plt.subplots()
        contourf_ = ax.contourf(Y, X, thick_rot, levels=np.linspace(vmin, vmax, 30), extend='max')
        cbar = fig.colorbar(contourf_, ticks=range(vmin, vmax + 3, 3))
        name = ImageProcessor.get_filename(f)
        image_path = os.path.join(savepath, name + ".png")
        plt.savefig(image_path)
        plt.close()


def get_overall_image(path, savepath):
    _ = ImageProcessor.get_overall_image(path, savepath)


def get_overall_images(rootpath):
    files = glob.glob(rootpath + "/*.czi")
    output = os.path.join(rootpath, "result/all_image")
    os.makedirs(output, exist_ok=True)
    rst = {}
    for f in tqdm(files):
        name = ImageProcessor.get_filename(f)
        savepath = os.path.join(output, name)
        get_overall_image(f, savepath)


def get_cross_info(rootpath, chans, savepath):
    files = glob.glob(rootpath + "/*.czi")
    file_headers = ["Name", "Volume", "Area", "Thickness"]
    with open(savepath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(file_headers)
        for i, f in enumerate(files):
            s = "(" + str(i) + "/" + str(len(files)) + ")"
            name = ImageProcessor.get_filename(f)
            print("\n==================", s, "=========================\n")
            print("processing: ", name)
            dt = ImageProcessor.get_cross(f, chans)
            im2 = dt.max(axis=0)
            area = np.count_nonzero(im2)
            volume = np.sum(dt)
            thickness = volume / area
            writer.writerow([name, volume, area, thickness])


def get_cross_info_from_results(rootpath, chans, savepath, rst_files):
    if len(chans) != len(rst_files):
        print("Number of result files is not match!")
        return
    files = glob.glob(rootpath + "/*.czi")
    file_headers = ["Name", "Volume", "Area", "Thickness"]
    rsts = {}
    for i, chan in enumerate(chans):
        rst_path = rst_files[i]
        with open(rst_path, 'rb') as handle:
            rst = pickle.load(handle)
            # rst[name] = (a, h, I, thickness_single, thickness_down, thickness_up, dims, thresh_min, chan, cell_chan)
            rsts[chan] = rst
    with open(savepath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(file_headers)
        for i, f in enumerate(files):
            s = "(" + str(i) + "/" + str(len(files)) + ")"
            name = ImageProcessor.get_filename(f)
            print("\n==================", s, "=========================\n")
            print("processing: ", name)
            th_mins = {}
            for ch in chans:
                rst = rsts[ch]
                if name in rst:
                    th_mins[ch] = rst[name][7]
                    if ch != rst[name][8]:
                        print("Waring: channel number is not match in ", f)
            dt = ImageProcessor.get_cross_with_min_threshold(czi_path=f, chans=chans, th_mins=th_mins)
            im2 = dt.max(axis=0)
            area = np.count_nonzero(im2)
            volume = np.sum(dt)
            thickness = volume / area
            writer.writerow([name, volume, area, thickness])


def slice_image(rootpath, brightness_factor=[8, 16, 1], target_chan=1):
    files = glob.glob(rootpath + "/*.czi")
    output = os.path.join(rootpath, "result/thickness_slices")
    # os.makedirs(output, exist_ok=True)
    for f in files:
        name = ImageProcessor.get_filename(f)
        savepath = os.path.join(output, name)
        filename = "cell_count_map.tiff"
        map_file = os.path.join(savepath, filename)
        os.makedirs(savepath, exist_ok=True)

        cells, _ = ImageProcessor.count_cell(f, channel=target_chan, save_path=map_file)
        cells_image = cv.imread(map_file)
        cells_image_labeled = cells_image.copy()
        height = cells_image.shape[0]
        width = cells_image.shape[1]
        slices_x = []
        slices_y = []
        for i, cell in enumerate(cells):
            span = cell[1]
            x = int(0.5 * (span[0] + span[1]))
            y = int(0.5 * (span[2] + span[3]))
            slices_x.append(x)
            slices_y.append(y)
            color = (0, 0, 255)
            thickness = 2
            cells_image = cv.line(cells_image, (0, y), (width, y), color, thickness)
            cells_image = cv.line(cells_image, (x, 0), (x, height), color, thickness)
            cv.putText(cells_image_labeled, str(i + 1), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        filename_cells_image = os.path.join(savepath, "cell_count_map_cut.tiff")
        filename_cells_image_labeled = os.path.join(savepath, "cell_count_map_labeled.tiff")
        cv.imwrite(filename_cells_image_labeled, cells_image_labeled)
        cv.imwrite(filename_cells_image, cells_image)
        ImageProcessor.save_figs(im_path=f, savepath=savepath, slice_dir="X",
                                 slice_idx=slices_x, brightness_factor=brightness_factor)
        ImageProcessor.save_figs(im_path=f, savepath=savepath, slice_dir="Y",
                                 slice_idx=slices_y, brightness_factor=brightness_factor)


def split_thickness(rootpath):
    files = glob.glob(rootpath + "/*.czi")
    output = os.path.join(rootpath, "result/thickness_slices")
    # os.makedirs(output, exist_ok=True)
    for f in files:
        name = ImageProcessor.get_filename(f)
        savepath = os.path.join(output, name)
        filename = "cell_count_map.tiff"
        map_file = os.path.join(savepath, filename)
        os.makedirs(savepath, exist_ok=True)


def varify_files(rootpath):
    files = glob.glob(rootpath + "/*.czi")
    error_file = os.path.join(rootpath, "error.log")
    print("Verifying...")
    crop = []
    flag = True
    for f in tqdm(files):
        try:
            ImageProcessor.get_data(f)
        except:
            with open(error_file, 'a') as s:
                s.write(f + " is not valid")
                flag = False
                crop.append(f)
    if flag:
        print("Verify completed, no bad file found!")
    else:
        print("Verify completed, ", str(len(crop)), " file not valid!")
        print("Check logfile at: ", error_file)
    return flag


def swap_channel(rootpath, order, suffix=".tiff"):
    files = glob.glob('**/*' + suffix, root_dir=rootpath, recursive=True)
    for f in tqdm(files):
        filepath = os.path.join(rootpath, f)
        img = cv.imread(filepath)
        img_new = np.zeros(np.shape(img), dtype=img.dtype)
        img_new[:, :, 0] = img[:, :, order[0]]
        img_new[:, :, 1] = img[:, :, order[1]]
        img_new[:, :, 2] = img[:, :, order[2]]
        # cv.imshow("000", img)
        # cv.imshow("aaa", img_new)
        cv.imwrite(filepath, img_new)


def make_dir(root, is_nc=False):
    savepath = os.path.join(root, "result")
    all_image_path = os.path.join(savepath, "all_image")
    thickness_map_path = os.path.join(savepath, "thickness_map")
    slice_image_path = os.path.join(savepath, "thickness_slices")
    binarize_img = os.path.join(savepath, "binarize_images")
    nc_image = os.path.join(savepath, "negative_control")
    os.makedirs(all_image_path, exist_ok=True)
    os.makedirs(thickness_map_path, exist_ok=True)
    os.makedirs(slice_image_path, exist_ok=True)
    os.makedirs(binarize_img, exist_ok=True)
    if is_nc:
        os.makedirs(nc_image, exist_ok=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
    # datapath = filedialog.askdirectory(initialdir='.')
    # savefile = "rst_layer.pickle"
    # make_dir(datapath)
    # treat_raw_data(datapath, savepath=savefile, overlap="layer")    # treat original data, saved in pickle file
    # group_analyse(datapath, savefile)    # compare area and thickness result between different group
    # get_thickness_image(datapath)  # convert thickness map to .dat file, for tecplot
    # get_overall_images(datapath)  # get overlapped image
    # ImageProcessor.save_figs(path, "./result/image/")    # save image slice separately
    # path = "D:\\pao\\Data\\new\\A005F2I1.czi"
    # path = "C:\\Users\\Administrator\\Downloads\\A005F2I1.czi"
    # rst = ImageProcessor.get_dim(path)
    # print(rst)
    # path = "/Users/nianhong/Downloads/A005F2I1.czi"
    # ImageProcessor.divide_cell(path, target_chan=0, cell_chan=1)
