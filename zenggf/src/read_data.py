# coding: utf-8
import re


def read_data(path):
    with open(path, encoding='utf-8') as f:
        data = [line.strip() for line in f.readlines()]
    data = list(filter(lambda x:clean(x), data))
    return data


def clean(line):
    if not line:
        return False
    result = re.findall(r'[a-zA-Z]+', line)
    if result:
        return False
    return True


def save_data(data, save_path):
    data = '\n'.join(data)
    with open(save_path, 'w', encoding='utf_8') as f:
        f.write(data)