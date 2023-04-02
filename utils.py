import argparse

'''
tool functions:
1. argparse for experiment (main.py)
'''


def hadle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", type=int, help="global round")
    args = parser.parse_args()
    return args