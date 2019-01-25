#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:08:36 2018

@author: qinyachen
"""

path=('/Users/qinyachen/Documents/514/email-Eu-core-department-labels.txt')

f = open(path, 'r')
lines = f.readlines()
f.close()

communites={}
value=[]

output=open('/Users/qinyachen/Documents/514/email-groudtruth.txt','w')

for line in lines:
    N1,N2=line.split()
    if N2 not in communites:
        communites[N2]=[N1]
    else:
        communites[N2].append(N1)

for line in communites.values():
    i=0
    for a in line:
        i=i+1
        output.write(a)
        if(i<len(line)):
            output.write(" ")
    output.write("\n")

output.close()