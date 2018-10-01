import os
import numpy as np
import dlib
from PIL import Image
import urllib
from StringIO import StringIO
from zipfile import ZipFile
import bz2
from sklearn.datasets import fetch_lfw_people

def create_lfw_db():
    lfw_people = fetch_lfw_people(resize=1,slice_=(slice(70, 250-30, None), slice(50, 250-50, None)))
    
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    filepath  = urllib.urlretrieve(url)[0]
    zipfile = bz2.BZ2File(filepath)
    data = zipfile.read()
    newfilepath = filepath[:-4]
    open(newfilepath, 'wb').write(data)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(newfilepath)
   
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
   
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    filepath  = urllib.urlretrieve(url)[0]
    zipfile = bz2.BZ2File(filepath)
    data = zipfile.read()
    newfilepath = filepath[:-4]
    open(newfilepath, 'wb').write(data)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(newfilepath)
   
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
    if not os.path.exists(dest+'mean/'):
        os.makedirs(dest+'mean/')
    
    face_db = dataset['images']
    for i in range(face_db.shape[0]):
        Image.fromarray(face_db[i]).save(dest+str(i+1)+'.pgm')
    Image.fromarray(np.mean(face_db,0).astype(np.uint8)).save(dest+'mean/mean.pgm')
   
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
        with open(dest+'mean/mean.vtk','w') as vtk:
            for item in file_vtk:
                vtk.write("%s\n" % item)