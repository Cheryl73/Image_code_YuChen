import numpy as np


class Node:
    def __init__(self, name, val=None):
        self.name = name
        self.val = val
        self.children = {}

    def add_child(self, child):
        self.children[child.name] = child

    def get_val(self):
        return self.val

    def refresh_val(self):
        if not self.children:
            return self.val
        rst = []
        for child in self.children.values():
            value = child.refresh_val()
            rst.append(value)
        average = np.mean(rst)
        self.set_val(average)
        return average

    def set_val(self, val):
        self.val = val


class data_tree:
    def __init__(self):
        self.rt = Node("Root")

    def add_data(self, key, val):
        sections = self.get_name_section(key)
        cur = self.rt
        for name in sections:
            if name in cur.children:
                cur = cur.children[name]
            else:
                nd = Node(name)
                cur.add_child(nd)
                cur = nd
        cur.set_val(val)
        self.refresh()

    def refresh(self):
        _ = self.rt.refresh_val()

    def get_data(self, key):
        sections = self.get_name_section(key)
        cur = self.rt
        for s in sections:
            cur = cur.children.get(s)
            if not cur:
                print(key, "not found!")
                return None
        return cur.val

    def get_height(self):
        def rec(rt):
            if not rt.children:
                return 1
            else:
                cur = 0
                for child in rt.children.values():
                    l = rec(child)
                    if l > cur:
                        cur = l
                return l + 1

        return rec(self.rt) - 1

    def print_data(self, level):
        height = self.get_height()

        def rec(rt, i, target, name):
            if i == target or i == height:
                print(name, rt.val)
                return
            else:
                for child in rt.children:
                    nd = rt.children[child]
                    rec(nd, i + 1, target, name + child)
            return

        rec(self.rt, 0, target=level, name='')

    @staticmethod
    def treat_name(key):
        syb = []
        num = []
        cur = ''
        digit = False
        for s in key:
            if s.isdigit():
                if digit:
                    cur += s
                else:
                    syb.append(cur)
                    digit = True
                    cur = s
            else:
                if digit:
                    cur = str(int(cur))
                    num.append(cur)
                    digit = False
                    cur = s
                else:
                    cur += s
        if digit:
            cur = str(int(cur))
            num.append(cur)
        return syb, num

    @staticmethod
    def get_name_section(key):
        syb, num = data_tree.treat_name(key)
        if len(syb) != len(num):
            print("Warning: Section number not match, check filename format!")
            # return
        rst = []
        for i, s in enumerate(syb):
            name = s + num[i]
            rst.append(name)
        return rst

    def print_data_recursive(self):
        def rec(rt: Node, curname):
            if not rt.children:
                print(curname, rt.val)
                return
            else:
                for child in rt.children:
                    nd = rt.children[child]
                    rec(nd, curname + child)
            if rt.name == 'Root':
                return
            print(curname, rt.val)
            return
        rec(self.rt, '')


class data_tree_cmb(data_tree):
    def __init__(self, data_names):
        super().__init__()
        self.dataNames = data_names




if __name__ == "__main__":
    s1 = 'skj_lhjk-%+-@#g$%-)k100k001n'
    val1 = 500
    s2 = 'skj_lhjk-%+-@#g$%-)k200k001n1'
    val2 = 200
    s3 = 'a1b'
    val3 = 400
    dt = data_tree()
    dt.add_data(s1, val1)
    dt.add_data(s2, val2)
    dt.add_data(s3, val3)
    # a = dt.get_data('s100')
    # print(a)
    # l = dt.get_height()
    # dt.print_data(2)
    # a = 1
    dt.print_data_recursive()