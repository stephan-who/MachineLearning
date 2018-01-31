# encoding: utf-8
"""
@author: gongxianglin
@contact: deamoncao100@gmail.com
@software: garner
@file: tree.py
@time: 2018/1/31 17:29
@desc:
"""
from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]


def stroeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)



