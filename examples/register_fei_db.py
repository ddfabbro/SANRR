#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:54:50 2017

@author: davi
"""
import numpy as np
import glob
import sys
sys.path.append('../source')
from sanrr import MyMIRTK
from mydata import create_fei_db

fei_db = create_fei_db()

configuration_file = '../mirtk_folder/register-2d-face-landmarks.cfg'
solver = MyMIRTK([],2,True,configuration_file)

solver.setPCA(fei_db['images'],.8)

for name in glob.glob('../mirtk_folder/im/*.pgm'):
    name = name[19:][:-4]
    files = {'ref_im': '../mirtk_folder/im/mean.pgm',
             'ref_vtk': '../mirtk_folder/im/mean.vtk',
             'mov_im': '../mirtk_folder/im/'+name+'.pgm',
             'mov_vtk': '../mirtk_folder/im/'+name+'.vtk',
             'out_im': '../mirtk_folder/transformed/'+name+'.pgm',
             'out_vtk': '../mirtk_folder/transformed/'+name+'.vtk',
             'dofs': '../mirtk_folder/dofs/b__a.dof.gz'}
    
    np.random.seed(1)
    solver.krigeRegister(files,11,5)
