#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:11:20 2017

@author: davi
"""
import numpy as np
from pyKriging.krige import kriging

class MyKriging(kriging):
    def __init__(self,*args,**kwargs):
        kriging.__init__(self,*args,**kwargs)
    def kdata(self):
        # Create a set of data to plot
        plotgrid = 61
        x = np.linspace(0, 1, num=plotgrid)
        y = np.linspace(0, 1, num=plotgrid)
        X, Y = np.meshgrid(x, y)
        # Predict based on the optimized results
        zs = np.array([self.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        #Calculate errors
        zse = np.array([self.predict_var([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Ze = zse.reshape(X.shape)
        #Sample point
        spx = (self.X[:,0] * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
        spy = (self.X[:,1] * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0] 
        return X,Y,Z,Ze,spx,spy