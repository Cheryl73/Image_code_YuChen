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
    def get_thickness(czi_path, method="count", chan=0):
        img = AICSImage(czi_path)
        dim = img.dims
        thickness = np.zeros([dim.Y, dim.X])
        im2 = ImageProcessor.binarize_Z(czi_path, chan=chan)
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

        # ImageProcessor.showImage('Watershed', markers, 'rgb')
        # print(len(set(markers.flatten()))-1)
        # print(np.max(markers) - 1)

        if save_path:
            matplotlib.image.imsave(save_path, markers)

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

        return rst

    @staticmethod
    def get_filename(czi_path):
        name = czi_path.split("\\")[-1]
        name = name.split(".")[0]
        return name

    @staticmethod
    def binarize_Z(czi_path, chan=0):
        img = AICSImage(czi_path)
        dim = img.dims
        rst = np.zeros([dim.Z, dim.Y, dim.X])
        for z in range(0, dim.Z):
            im = img.get_image_data("YX", C=chan, T=0, Z=z)
            # img_scaled = cv.normalize(im, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)
            # cv.imshow('im_scaled', img_scaled)
            # im_8 = im.astype(dtype='uint8')
            # dst = cv.equalizeHist(im_8)
            # cv.imshow('aaa', im_8)
            # cv.imshow('bbb', dst)
            th, im2 = cv.threshold(im, 0, 1, cv.THRESH_OTSU)
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
            rst[z, :, :] = im2
        return rst

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
            return 0, 0, 0, 0
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
    def statistic_ans(czi_path, overlap="layer", chan=0):
        '''
        :param overlap: overlap method: maximum or binarize by layer
        :param czi_path: file path of .czi image
        :param chan: channel number to analysis
        :return:
        '''
        print("Now treat: ", czi_path)
        # img = AICSImage(czi_path)
        # dim = img.dims
        if overlap == "max":
            im_all = ImageProcessor.get_overall_Z_max(czi_path, chan=chan)
            # cv.imshow("lalala", im_all)
            th, im2 = cv.threshold(im_all, 0, 1, cv.THRESH_OTSU)
            print("threshold for all image: ", th)
            area = np.count_nonzero(im2)
        else:
            bi_z = ImageProcessor.binarize_Z(czi_path, chan=chan)
            im2 = bi_z.max(axis=0)
            area = np.count_nonzero(im2)

        name = ImageProcessor.get_filename(czi_path)
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

        thickness = ImageProcessor.get_thickness(czi_path, method="count", chan=chan)
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
        return area, thickness_mask, intensity

    @staticmethod
    def mask_thickness_by_area(thickness, area_bi):
        return thickness * area_bi

    @staticmethod
    def treat_raw_data(data_directory, filepath, overlap="layer", chan=0):
        files = glob.glob(data_directory + "/*.czi")
        rst = {}
        for f in files:
            a, h, I = ImageProcessor.statistic_ans(f, overlap=overlap, chan=chan)
            name = ImageProcessor.get_filename(f)
            rst[name] = (a, h, I)
        with open(filepath, 'wb') as handle:
            pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data is treated and save as " + filepath)

    @staticmethod
    def get_cross(czi_path, chans):
        rst = 1
        for chan in chans:
            bi_z = ImageProcessor.binarize_Z(czi_path, chan=chan)
            v = np.count_nonzero(bi_z)
            print("channel ", chan, " Volume:", v)
            rst = rst * bi_z
        s = np.sum(rst)
        print("Number of overlapping pixels: ", s)
        return s


def treat_raw_data(data_path, savepath, overlap="layer", chan=0):
    ImageProcessor.treat_raw_data(data_path, savepath, overlap, chan=chan)


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


def group_analyse(filepath):
    with open(filepath, 'rb') as handle:
        rst = pickle.load(handle)
    names = rst.keys()
    if not names:
        print("No result found, check result file!")
        return
    info, order = get_name_info(names)
    concentrations = info[order[0]]
    samples = info[order[1]]
    images = info[order[2]]

    area_c = []
    h_c = []
    v_c = []
    pre = ''
    group = {}
    for c in concentrations:
        group[c] = {}
        area_s = []
        h_s = []
        v_s = []
        for s in samples:
            group[c][s] = {}
            area_i = []
            h_i = []
            v_i = []
            for i in images:
                info = [c, s, i]
                key = get_file_key(info, order=order, pre=pre)
                if key not in rst:
                    print(key, " not found")
                    continue
                cur = rst[key]
                # print(key, " found")
                area_ = cur[0]
                h_ = np.sum(cur[1]) / area_
                v_ = area_ * h_
                group[c][s][i] = (area_, h_, v_)
                area_i.append(area_)
                h_i.append(h_)
                v_i.append(v_)
                print(c, s, i, area_, h_, v_)
            area_s.append(np.average(area_i))
            h_s.append(np.average(h_i))
            v_s.append(np.average(v_i))
            group[c][s]["mean"] = (np.average(area_i), np.average(h_i), np.average(v_i))
            print(c, s, "mean ", group[c][s]["mean"])
        area_c.append(np.average(area_s))
        h_c.append(np.average(h_s))
        v_c.append(np.average(v_s))
        group[c]["mean"] = (np.average(area_s), np.average(h_s), np.average(v_s))
        print(c, "mean ", group[c]["mean"])

    rst_c = [group[c]["mean"] for c in concentrations]
    rst_area = [x[0] for x in rst_c]
    rst_t = [x[1] for x in rst_c]
    rst_v = [x[2] for x in rst_c]
    fig = plt.figure(1)
    plt.plot(concentrations, rst_area, marker="o")
    plt.title("area")
    fig = plt.figure(2)
    plt.plot(concentrations, rst_t, marker="o")
    plt.title("thickness")
    fig = plt.figure(3)
    plt.plot(concentrations, rst_v, marker="o")
    plt.title("Volume")
    plt.show()


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
    # im_G = ImageProcessor.get_overall_Z_max(path, 0)
    # im_B = ImageProcessor.get_overall_Z_max(path, 1)
    # im_R = 0 * im_G
    # im_G0 = cv.merge([im_R, im_G, im_R])
    # im_B0 = cv.merge([im_B, im_R, im_R])
    # im_all = cv.merge([im_B, im_G, im_R])
    # path_all = savepath + "_all.tiff"
    # path_B = savepath + "_B.tiff"
    # path_G = savepath + "_G.tiff"
    # cv.imwrite(path_all, im_all)
    # cv.imwrite(path_B, im_B0)
    # cv.imwrite(path_G, im_G0)


def get_overall_images(rootpath):
    files = glob.glob(rootpath + "/*.czi")
    output = os.path.join(rootpath, "result/all_image")
    os.makedirs(output, exist_ok=True)
    rst = {}
    for f in tqdm(files):
        name = ImageProcessor.get_filename(f)
        savepath = os.path.join(output, name)
        get_overall_image(f, savepath)


def get_cross_info(rootpath, chans):
    files = glob.glob(rootpath + "/*.czi")
    for f in files:
        name = ImageProcessor.get_filename(f)
        print("\n===========================================\n")
        print("processing: ", name)
        ImageProcessor.get_cross(f, chans)


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

        cells = ImageProcessor.count_cell(f, channel=target_chan, save_path=map_file)
        cells_image = cv.imread(map_file)
        height = cells_image.shape[0]
        width = cells_image.shape[1]
        slices_x = []
        slices_y = []
        for cell in cells:
            span = cell[1]
            x = int(0.5 * (span[0] + span[1]))
            y = int(0.5 * (span[2] + span[3]))
            slices_x.append(x)
            slices_y.append(y)
            color = (0, 0, 255)
            thickness = 2
            cells_image = cv.line(cells_image, (0, y), (width, y), color, thickness)
            cells_image = cv.line(cells_image, (x, 0), (x, height), color, thickness)
        filename_cells_image = os.path.join(savepath, "cell_count_map_cut.tiff")
        cv.imwrite(filename_cells_image, cells_image)
        ImageProcessor.save_figs(im_path=f, savepath=savepath, slice_dir="X",
                                 slice_idx=slices_x, brightness_factor=brightness_factor)
        ImageProcessor.save_figs(im_path=f, savepath=savepath, slice_dir="Y",
                                 slice_idx=slices_y, brightness_factor=brightness_factor)


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


def make_dir(root):
    savepath = os.path.join(root, "result")
    all_image_path = os.path.join(savepath, "all_image")
    thickness_map_path = os.path.join(savepath, "thickness_map")
    slice_image_path = os.path.join(savepath, "thickness_slices")
    binarize_img = os.path.join(savepath, "binarize_images")
    os.makedirs(all_image_path, exist_ok=True)
    os.makedirs(thickness_map_path, exist_ok=True)
    os.makedirs(slice_image_path, exist_ok=True)
    os.makedirs(binarize_img, exist_ok=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # datapath = filedialog.askdirectory(initialdir='.')
    # savefile = "rst_layer.pickle"
    # make_dir(datapath)
    # treat_raw_data(datapath, savepath=savefile, overlap="layer")    # treat original data, saved in pickle file
    # group_analyse(datapath, savefile)    # compare area and thickness result between different group
    # get_thickness_image(datapath)  # convert thickness map to .dat file, for tecplot
    # get_overall_images(datapath)  # get overlapped image
    # ImageProcessor.save_figs(path, "./result/image/")    # save image slice separately
    path = "C:\\Users\\Administrator\\Downloads\\czi-20231120T070211Z-001\\czi\\H2F1I1.czi"
    rst = ImageProcessor.get_overall_mean(path, channel=0)
    print(rst)
    # rst = ImageProcessor.get_cross(path, [0,1])
    # print(rst)
    # path = "D:\pao\Data\\new2"
    # order = [2,1,0]
    # swap_channel(path, order)
    # cells = ImageProcessor.count_cell(path, 1)
    # slices_x = []
    # slices_y = []
    # for cell in cells:
    #     span = cell[1]
    #     x = int(0.5 * (span[0] + span[1]))
    #     y = int(0.5 * (span[2] + span[3]))
    #     slices_x.append(x)
    #     slices_y.append(y)
    # ImageProcessor.save_figs(im_path=path, savepath="Data/new/result/thickness_slices/", slice_dir="X", slice_idx=slices_x, brightness_factor = 6)
    # ImageProcessor.save_figs(im_path=path, savepath="Data/new/result/thickness_slices/", slice_dir="Y", slice_idx=slices_y, brightness_factor = 6)
