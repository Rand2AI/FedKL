# -*-coding:utf-8-*-
import os
import json
import sys

__CRNN_CONFIG__ = None


def get_config(file: str = None) -> dict:
    global __CRNN_CONFIG__
    if file is None:
        file = os.path.dirname(os.path.realpath(__file__)) + '/config.json'
    else:
        file += '/config.json'
    if __CRNN_CONFIG__ is None:
        try:
            with open(file, 'r') as fid:
                __CRNN_CONFIG__ = json.load(fid)
        except:
            print('Unexpected Error:', sys.exc_info())
    return __CRNN_CONFIG__


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass