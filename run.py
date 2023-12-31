import image_pao
import tkinter as tk
from tkinter import filedialog
import os


def getfile(title = "select", types = [], action = "open", initial = "result"):
    root = tk.Tk()
    root.withdraw()
    if action == "save":
        file_path = filedialog.asksaveasfilename(title=title, filetypes=[("", types)], initialfile = initial)
    elif action =="path":
        file_path = filedialog.askdirectory(title=title)
    else:
        file_path = filedialog.askopenfilename(title=title, filetypes=[("", types)])
    return file_path

class MainMenu:

    def __init__(self) -> None:
        self.root = None
        # self.run()

    def run(self):
        current_menu = self.root
        if not current_menu:
            print("this is a empty menu...")
            return
        while True:
            try:
                if current_menu.todo:
                    current_menu.func()
                    if current_menu.child_num == 0:
                        current_menu = current_menu.parent
                current_menu.show_menu()
                opt = input("Enter the choice:\n -- Ctrl+c or 0: exit \n -- enter: return to previous \n -- m: return "
                            "to main menu)\n:")
                if opt == "":
                    if current_menu.parent is None:
                        continue
                    current_menu = current_menu.parent
                elif opt == "0":
                    break
                elif opt == "m":
                    current_menu = self.root
                elif int(opt) > current_menu.child_num:
                    print("Input out of index, please retry!")
                else:
                    current_menu = current_menu.to_next(int(opt))
            except KeyboardInterrupt as e:
                break
        self._exit()

    def _exit(self):
        pass


class Menu:
    def __init__(self, title: str = "main") -> None:
        self.title = title
        self.children = []
        self.parent = None
        self.func = None
        self.child_num = 0
        self.todo = 0
        self.show_func = None
        self.show_do = 0

    def _set_parent(self, menu):
        self.parent = menu

    def add_child(self, menu):
        menu._set_parent(self)  # set parent
        self.children.append(menu)
        self.child_num += 1

    def show_menu(self):
        print("\n======================\n")
        for i in range(len(self.children)):
            print("[%s]  %s" % ((i + 1), self.children[i].title))
        print("\n")
        if self.show_do:
            self.show_func()

    def to_next(self, index: int):
        index = int(index) - 1
        return self.children[index]

    def do(self, func):
        self.func = func
        self.todo = 1

    def show_and_do(self, show_func):
        self.show_func = show_func
        self.show_do = 1


class appMenu(MainMenu):
    def __init__(self):
        super().__init__()
        self.datapath = ''
        self.rstfile = ''
        self.brightness_factor = None
        # self.datapath = getfile("Choose data directory", action="path")
        # self.rstfile = getfile("Select result file")
        self.build()

    def build(self):
        menu = Menu()
        self.root = menu

    # level one
        set_path = Menu("Select test result path")
        set_file = Menu("Save result file as")
        select_rst = Menu("Select result file")
        treat_rawdata = Menu("Treat raw data")
        slice_image = Menu("Get thickness image")
        set_factor = Menu("Set brightness factor")
        group_analyse = Menu("Analyse group result")
        get_thickness_image = Menu("Get thickness map (.dat file)")
        get_overall_images = Menu("Get overall images")
        cross = Menu("Statistical analysis of overlapping pixels")
        swap = Menu("Swap color of channels")

        menu.add_child(get_overall_images)
        menu.add_child(treat_rawdata)
        menu.add_child(group_analyse)
        menu.add_child(slice_image)
        menu.add_child(get_thickness_image)
        menu.add_child(set_factor)
        menu.add_child(set_path)
        menu.add_child(set_file)
        menu.add_child(select_rst)
        menu.add_child(cross)
        menu.add_child(swap)



        @menu.do
        def title():
            print("======================\n")
            print("This program is created by Yu")
            print("Yu is invincible!!!!")
            for row in range(6):
                for col in range(7):
                    if (row == 0 and col % 3 != 0) or (row == 1 and col % 3 == 0) or (row - col == 2) or (
                            row + col == 8):
                        print('*', end=' ')
                    else:
                        print(' ', end=' ')
                print()

        @menu.show_and_do
        def show_path():
            print("\n======================\n")
            if self.datapath:
                print("Data path: ", self.datapath)
            else:
                print("Data path not set")
            if self.rstfile:
                print("Analysis result file: ", self.rstfile)
            else:
                print("Result file not set")

        @treat_rawdata.do
        def treat_rawdata():
            channel = input("Select channel for analysis(0, 1, 2, defult = 0): ", ) or "0" # original chan order: BGR
            channel = int(channel)
            if self.datapath:
                datapath = self.datapath
            else:
                datapath = getfile("Choose data directory", action="path")
            image_pao.make_dir(datapath)
            file_path = getfile("Save as: ", action="save", types=['.yu'], initial="result_chan" + str(channel) + ".yu")
            if not file_path or not datapath:
                print("No path setting!")
                return
            image_pao.treat_raw_data(datapath, file_path, chan=channel)
            return datapath, file_path

        @get_thickness_image.do
        def get_thickness_image():
            if self.datapath:
                datapath = self.datapath
            else:
                datapath = getfile("Choose output directory", action="path")
            if self.rstfile:
                filepath = self.rstfile
            else:
                filepath = getfile("Select result file", action="open")

            if not filepath or not datapath:
                print("No path setting!")
                return
            image_pao.get_thickness_image(datapath, filepath)

        @get_overall_images.do
        def get_overall_images():
            if self.datapath:
                datapath = self.datapath
            else:
                datapath = getfile("Choose data directory", action="path")
            if not datapath:
                print("No path setting!")
                return
            image_pao.get_overall_images(datapath)

        @group_analyse.do
        def group_analyse():
            if self.rstfile:
                filepath = self.rstfile
            else:
                filepath = getfile("Select result file", action="open")

            if not filepath:
                print("No path setting!")
                return
            image_pao.group_analyse2(filepath)

        @set_path.do
        def set_path():
            datapath = getfile("Choose data directory", action="path")
            self.datapath = datapath
            return datapath

        @set_file.do
        def set_file():
            file = getfile("Save as: ", action="save")
            self.rstfile = file
            return file

        @select_rst.do
        def select_file():
            file = getfile("Select result file", types=[".pickle"])
            self.rstfile = file
            return file

        @slice_image.do
        def slice_image():
            if self.datapath:
                datapath = self.datapath
            else:
                datapath = getfile("Choose data directory", action="path")
            if not datapath:
                print("No path setting!")
                return
            target_chan = input("Input target channel number (0/1/2): ", ) or "1"
            target_chan = int(target_chan)
            if target_chan not in (0,1,2):
                print("Channel number incorrect!")
                return
            if self.brightness_factor:
                image_pao.slice_image(datapath, brightness_factor=self.brightness_factor, target_chan=target_chan)
            else:
                image_pao.slice_image(datapath, target_chan=target_chan)

        @set_factor.do
        def set_factor():
            factor = input("Input the brightness factor for B-G-R channel, separated by commas (e.g. 8,16,1) (default)")
            f = factor.split(",")
            rst = []
            for item in f:
                rst.append(int(item))
            self.brightness_factor = rst


        @cross.do
        def cross():
            if self.datapath:
                datapath = self.datapath
            else:
                datapath = getfile("Choose data directory", action="path")
            if not datapath:
                print("No path setting!")
                return
            factor = input("Input the selected channels, separated by commas (e.g. 0, 1) (default)")
            f = factor.split(",")
            chans = []
            for item in f:
                chans.append(int(item))
            image_pao.get_cross_info(datapath, chans)

        @swap.do
        def swap():
            path = getfile("Choose root directory", action="path")
            s = input("input new channel order, separated by commas, out put order is BGR, i.e. (0,2,1 : BRG -> BGR) ")
            os = s.split(',')
            order = []
            for o in os:
                order.append(int(o))
            image_pao.swap_channel(path, order=order)

    def _exit(self):
        return 1

    def add_child_recursive(self, rt:Menu, item):
        if rt.child_num == 0:
            return
        else:
            exist = False
            for child in rt.children:
                self.add_child_recursive(child, item)
                if item == child:
                    exist = True
        if not exist:
            rt.add_child(item)
        return


if __name__ == "__main__":
    am = appMenu()
    am.run()








