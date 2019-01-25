#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:49:30 2018

@author: qinyachen
"""
def in_order(nodes, edges):
        # rebuild graph with successive identifiers
        nodes = list(nodes.keys())
        nodes.sort()
        i = 0
        nodes_ = []
        d = {}
        for n in nodes:
            nodes_.append(i)
            d[n] = i
            i += 1
        edges_ = []
        for e in edges:
            edges_.append(((d[e[0][0]], d[e[0][1]]), e[1]))
        return (nodes_, edges_,d)

path="/Users/qinyachen/Documents/514/ChG-Miner_miner-chem-gene.tsv"
f = open(path, 'r')
lines = f.readlines()
f.close()
nodes = {}
edges = []
i=0
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
    i=i+1
    if i>10:
        break

nodes_, edges_ ,d = in_order(nodes, edges)

for i in cls(nodes_,edges_):
    print(i)

