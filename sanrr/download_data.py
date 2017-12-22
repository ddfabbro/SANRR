#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:53:17 2017

@author: davi
"""
import numpy as np
import dlib
from PIL import Image
import urllib
from StringIO import StringIO
from zipfile import ZipFile
from sklearn.datasets import fetch_lfw_people

def create_lfw_db():
    lfw_people = fetch_lfw_people(resize=1,slice_=(slice(70, 250-30, None), slice(50, 250-50, None)))
   
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
   
    face_db = []
    landmarks_db=[]
   
    for i in range(lfw_people['images'].shape[0]):
        face = np.array(Image.fromarray(lfw_people['images'][i]).resize((100,100), Image.ANTIALIAS)).astype(np.uint8)
        dets = detector(face, 1)
        if len(dets) == 1:
            face_db.append(face)
            shape = predictor(face, dets[0])
            landmarks_db.append(np.array([[shape.part(i).x,shape.part(i).y] for i in range(68)]))
         
    return {'images': np.array(face_db),
            'landmarks': np.array(landmarks_db,dtype=np.float64)}

def create_fei_db():
    url1 = 'http://fei.edu.br/~cet/frontalimages_spatiallynormalized_part1.zip'
    url2 = 'http://fei.edu.br/~cet/frontalimages_spatiallynormalized_part2.zip'
    zip_files  = [urllib.urlretrieve(url1)[0],urllib.urlretrieve(url2)[0]]
    archive = [ZipFile(zip_files[0],'r'),ZipFile(zip_files[1],'r')]
   
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
   
    face_db = []
    landmarks_db=[]
   
    for name in archive[0].namelist()+archive[1].namelist():
        try:
            face = np.array(Image.open(StringIO(archive[0].read(name))).crop((0,50,250,300)))
        except:
            face = np.array(Image.open(StringIO(archive[1].read(name))).crop((0,50,250,300)))
         
        dets = detector(face, 1)
        if len(dets) == 1:
            face_db.append(face)
            shape = predictor(face, dets[0])
            landmarks_db.append(np.array([[shape.part(i).x,shape.part(i).y] for i in range(68)]))
            
    return {'images': np.array(face_db),
            'landmarks': np.array(landmarks_db,dtype=np.float64)}

def save_files(dataset,dest):
    face_db = dataset['images']
    for i in range(face_db.shape[0]):
        Image.fromarray(face_db[i]).save(dest+str(i+1)+'.pgm')
    Image.fromarray(np.mean(face_db,0).astype(np.uint8)).save(dest+'mean.pgm')
   
    landmarks_db = dataset['landmarks']
   
    header = ["# vtk DataFile Version 3.0","vtk output",
              "ASCII","DATASET POLYDATA","POINTS "+str(landmarks_db.shape[1])+" float"]
   
    for i in range(landmarks_db.shape[0]):
        file_vtk = []
        for line in header:
            file_vtk.append(line)
        for line in landmarks_db[i]:
            file_vtk.append(str(line[0]-face_db.shape[1]/2.)+' '+str(line[1]-face_db.shape[2]/2.)+' 0.0')                                         
            with open(dest+str(i+1)+'.vtk','w') as vtk:
                for item in file_vtk:
                    vtk.write("%s\n" % item)
  
    landmarks_db = np.mean(landmarks_db,0)             
    file_vtk = []
    for line in header:
        file_vtk.append(line)
    for line in landmarks_db:
        file_vtk.append(str(line[0]-face_db.shape[1]/2.)+' '+str(line[1]-face_db.shape[2]/2.)+' 0.0')                                         
        with open(dest+'mean.vtk','w') as vtk:
            for item in file_vtk:
                vtk.write("%s\n" % item)