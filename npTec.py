import numpy as np
from typing import TypeVar
ndarray = TypeVar('ndarray')
import time


def dimension_check(vrs, datas: list[ndarray], names:[list[str], list[str]]):
    num_var = len(vrs)
    var_shape = []
    for var in vrs:
        var_shape.append(len(var))
    var_shape = tuple(var_shape)
    num_data = len(datas)
    if num_data:
        data_shape = np.shape(datas[0])
        if data_shape == var_shape:
            shape_check = True
        else:
            print("data shape not match with variable shape, check dimension!")
            shape_check = False
    else:
        shape_check = True

    var_name_size = len(names[0])
    data_name_size = len(names[1])
    name_size = var_name_size + data_name_size
    if num_var + num_data == name_size:
        size_check = True
    else:
        print("Number of names not match with data and variable!")
        size_check = False
    return size_check and shape_check


def np2tec(vars, datas, names, filepath):
    if not dimension_check(vars, datas, names):
        print("Dimension check failed, check data!")
        return

    name_list = []
    axis_list = ["I", "J", "K"]
    for name in names[0]:
        name_list.append(name)
    for name in names[1]:
        name_list.append(name)

    var_num = len(names[0])
    data_num = len(names[1])
    item_num = len(name_list)

    with open(filepath, 'w') as f:
        f.write("VARIABLES = ")
        for vr in name_list:
            f.write("\"" + vr + "\", ")
        f.write("\n")
        f.write("ZONE T = \"Frame 1\", ")
        for i in range(0, len(vars)):
            f.write(axis_list[i] + "=" + str(len(vars[i])) + ", ")
        f.write("\n")

    def rec(i, values, index, f):
        if i >= var_num:
            cur_line = ''
            for value in values:
                s = "{:.8f}".format(value)
                cur_line += s + " "
            for data in datas:
                cur = data
                for k in index:
                    cur = cur[k]
                s = "{:.8f}".format(cur)
                cur_line += s + " "
            f.write(cur_line)
            f.write("\n")
        else:
            cur_var = vars[i]
            cur_values = values.copy()
            cur_index = index.copy()
            for j, cur in enumerate(cur_var):
                cur_values.append(cur)
                cur_index.append(j)
                rec(i + 1, cur_values, cur_index, f)
                cur_index.pop()
                cur_values.pop()
    with open(filepath, 'a') as f:
        rec(0, [], [], f)
    return


if __name__ == '__main__':
    n = 100
    x = np.linspace(0, n, n)
    y = np.linspace(0, n, n)
    h = np.zeros([n,n])
    k = np.ones([n,n])
    data = [h, k]
    names = [["x", "y"], ["h", "k"]]
    file = "try.dat"
    vrs = [x, y]
    np2tec(vrs, data, [["x", "y"],["h", "k"]], file)

