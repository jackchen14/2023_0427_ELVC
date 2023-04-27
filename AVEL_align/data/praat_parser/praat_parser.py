import re
import argparse
import json
from collections import OrderedDict
import os
from pathlib import Path
from pypinyin import pinyin, lazy_pinyin


# Reference from https://github.com/hschen0712/textgrid-parser/blob/master/parse_textgrid.py
class TextGrid(object):
    def __init__(self, text):
        self.text = text
        self.line_count = 0
        self._get_type()
        self._get_time_intval()
        self._get_size()
        self.tier_list = []
        self._get_item_list()

    def _extract_pattern(self, pattern, inc):
        """
        Parameters
        ----------
        pattern : regex to extract pattern
        inc : increment of line count after extraction
        Returns
        -------
        group : extracted info
        """
        try:
            group = re.match(pattern, self.text[self.line_count]).group(1)
            self.line_count += inc
        except AttributeError:
            raise ValueError("File format error at line %d:%s" % (self.line_count, self.text[self.line_count]))
        return group

    def _get_type(self):
        self.file_type = self._extract_pattern(r"File type = \"(.*)\"", 2)

    def _get_time_intval(self):
        self.xmin = self._extract_pattern(r"xmin = (.*)", 1)
        self.xmax = self._extract_pattern(r"xmax = (.*)", 2)

    def _get_size(self):
        self.size = int(self._extract_pattern(r"size = (.*)", 2))

    def _get_item_list(self):
        """Only supports IntervalTier currently"""
        for itemIdx in range(1, self.size + 1):
            tier = OrderedDict()
            item_list = []
            tier_idx = self._extract_pattern(r"item \[(.*)\]:", 1)
            tier_class = self._extract_pattern(r"class = \"(.*)\"", 1)
            if tier_class != "IntervalTier":
                raise NotImplementedError("Only IntervalTier class is supported currently")
            tier_name = self._extract_pattern(r"name = \"(.*)\"", 1)
            tier_xmin = self._extract_pattern(r"xmin = (.*)", 1)
            tier_xmax = self._extract_pattern(r"xmax = (.*)", 1)
            tier_size = self._extract_pattern(r"intervals: size = (.*)", 1)
            for i in range(int(tier_size)):
                item = OrderedDict()
                item["idx"] = self._extract_pattern(r"intervals \[(.*)\]", 1)
                item["xmin"] = self._extract_pattern(r"xmin = (.*)", 1)
                item["xmax"] = self._extract_pattern(r"xmax = (.*)", 1)
                item["text"] = self._extract_pattern(r"text = \"(.*)\"", 1)
                item_list.append(item)
            tier["idx"] = tier_idx
            tier["class"] = tier_class
            tier["name"] = tier_name
            tier["xmin"] = tier_xmin
            tier["xmax"] = tier_xmax
            tier["size"] = tier_size
            tier["items"] = item_list
            self.tier_list.append(tier)


    def toJson(self):
        _json = OrderedDict()
        _json["file_type"] = self.file_type
        _json["xmin"] = self.xmin
        _json["xmax"] = self.xmax
        _json["size"] = self.size
        _json["tiers"] = self.tier_list
        return json.dumps(_json, ensure_ascii=False, indent=2).encode("utf-8")



def parse_nl_textgrid(path, encoding="utf-8"):

    # Read in file
    with open(path, "r", encoding=encoding) as f:
        lines = f.read().splitlines()
        if "" in lines:
            lines.remove("")
        lines = [line.lstrip() for line in lines]
        
        i = 0
        while i < len(lines):
            if lines[i] == '"HANZI/wrd"': break
            i+=1
        i += 3
        n = int(lines[i])
        i += 1
        item_list = []
        for j in range(n):
            item = dict()
            item["xmin"] = str(lines[i+j*3])
            item["xmax"] = str(lines[i+j*3+1])
            item["text"] = lines[i+j*3+2][1] if len(lines[i+j*3+2]) == 3 else ""
            item_list.append(item)

        return item_list

def parse_el_textgrid(path):

    # Read in file
    with open(path, "r", encoding="utf-16") as f:
        lines = f.read().splitlines()
        if "" in lines:
            lines.remove("")
        lines = [line.lstrip() for line in lines]
        
        # parse
        textgrid = TextGrid(lines)

        return textgrid

def in_interval(x, y, intervals):
    for el_interval, nl_interval in intervals:
        if x >= el_interval[0] and x <= el_interval[1] and y >= nl_interval[0] and y <= nl_interval[1]:
            return True
    return False

def get_fplist(filepath, tar_ext='wav'):
    list_all = []
    for dirPath, dirNames, fileNames in os.walk(filepath):
        for n_itr, f in enumerate(fileNames):            
            cur_ext = Path(f).suffix[1:]
            if cur_ext == tar_ext:                
                list_all.append(Path(filepath, dirPath, f))
    return list_all


def print_label(list_data, title='el_borders'):

    print(f'label of {title}')
    for item in list_data:
        print(f'{item :.04f}')

def main(args):

    ### check data pair
    list_el_label = get_fplist(args.el_label_dir, tar_ext='Textgrid')
    list_nl_label = get_fplist(args.nl_label_dir, tar_ext='Textgrid')
    list_el_label.sort()
    list_nl_label.sort()
    
    el_db = Path(args.el_wav_dir)
    nl_db = Path(args.nl_wav_dir)

    n_pair=0
    list_pair = []
    for utt_idx in range(1, 321):
        cur_fn=f'*{utt_idx:03d}*'
        try: 
            temp_el=list(el_db.glob(cur_fn))
            temp_nl=list(nl_db.glob(cur_fn))
            if len(temp_el)==1 and len(temp_nl)==1:
                if (not temp_nl[0].match('*_w*')) and (not temp_el[0].match('*_w*')):
                    list_pair.append(f'{utt_idx:03d}')
                    n_pair+=1
        except:
            print(f'{cur_nl} is not found')

    ### parse label by utt_idx
    for utt_idx in list_pair:

        el_label_path=list(Path(args.el_label_dir).glob(f'*{utt_idx}*'))[0]
        nl_label_path=list(Path(args.nl_label_dir).glob(f'*{utt_idx}*'))[0]

        print(el_label_path)
        print(nl_label_path)

        try:
            el_label = parse_el_textgrid(el_label_path).tier_list[0]["items"]
            nl_label = parse_nl_textgrid(nl_label_path)
        except:
            print(f' el/nl label {utt_idx}')
            continue
    
        text = lazy_pinyin("".join([item["text"] for item in el_label if item["text"] != ""]))

        el_borders, el_xmins, el_xmaxs = [], [], []
        nl_borders, nl_xmins, nl_xmaxs = [], [], []

        ### for plot interval
        for item in el_label:
            if item["text"] != "":
                el_borders.append(float(item["xmin"]))
                el_borders.append(float(item["xmax"]))
                nl_xmins.append(float(item["xmin"]))
                nl_xmins.append(float(item["xmin"]))
                nl_xmaxs.append(float(item["xmax"]))
                nl_xmaxs.append(float(item["xmax"]))
        for item in nl_label:
            if item["text"] != "":
                el_xmins.append(float(item["xmin"]))
                el_xmins.append(float(item["xmin"]))
                el_xmaxs.append(float(item["xmax"]))
                el_xmaxs.append(float(item["xmax"]))
                nl_borders.append(float(item["xmin"]))
                nl_borders.append(float(item["xmax"]))

        el_ticks = list(set(nl_xmins+nl_xmaxs))
        nl_ticks = list(set(el_xmins+el_xmaxs))

        print_label(el_borders, title='el_borders')
        print_label(el_xmins, title='el_xmins')
        print_label(el_xmaxs, title='el_xmaxs')
        
        print_label(nl_borders, title='nl_borders')
        print_label(nl_xmins, title='nl_xmins')
        print_label(nl_xmaxs, title='nl_xmaxs')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--el_label_dir", type=str, default='../EL01/praat')
    parser.add_argument("--nl_label_dir", type=str, default='../NL01/praat')
    parser.add_argument("--el_wav_dir", type=str, default='../EL01/wav')
    parser.add_argument("--nl_wav_dir", type=str, default='../NL01/wav')
    args = parser.parse_args()
    
    main(args)
