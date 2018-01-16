#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:54:50 2017

@author: davi
"""
import sys
sys.path.append('..')

import numpy as np
import glob
from sanrr.register import SANRR
from sanrr.download_data import create_fei_db, save_files

fei_db = create_fei_db()
save_files(fei_db,'../mirtk_folder/im/')

configuration_file = '../mirtk_folder/register-2d-face-landmarks.cfg'
solver = SANRR([],2,True,configuration_file)

solver.setPCA(fei_db['images'],.8)

files = {'ref_im': '../mirtk_folder/im/mean/mean.pgm',
         'ref_vtk': '../mirtk_folder/im/mean/mean.vtk',
         'dofs': '../mirtk_folder/dofs/b__a.dof.gz'}

for name in glob.glob('../mirtk_folder/im/*.pgm'):
    name = name[19:][:-4]
    files['mov_im'] = '../mirtk_folder/im/'+name+'.pgm'
    files['mov_vtk'] = '../mirtk_folder/im/'+name+'.vtk'
    files['out_im'] = '../mirtk_folder/transformed/'+name+'.pgm'
    files['out_vtk'] = '../mirtk_folder/transformed/'+name+'.vtk'
    
    np.random.seed(0)
    solver.krigeRegister(files,11,5)