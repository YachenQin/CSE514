#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:18:53 2018

@author: qinyachen
"""

#import matplotlib.pyplot as plt
#import networkx as nx



f=open('/Users/qinyachen/Documents/514/amazon0302.txt','r')

lines = f.readlines()
f.close()
nodes = {}
edges = []
for line in lines:
    n = line.split()
    if not n:
        break
    nodes[n[0]] = 1
    nodes[n[1]] = 1
    w = 1
    if len(n) == 3:
        w = int(n[2])
    edges.append(((n[0], n[1]), w))

print(len(nodes))