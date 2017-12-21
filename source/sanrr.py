#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:56:44 2017

@author: davi
"""
import os
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pyKriging.krige import kriging
from pyDOE import lhs
 
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
         
   def myplot(self,k,i):
      fig = plt.figure()
      ax = fig.add_subplot(111)
      X,Y,Z,Ze,spx,spy=k.kdata()
      rcParams['contour.negative_linestyle'] = 'solid'
      plt.contour(X,Y,Z,10, cmap='gray')
      plt.scatter(spx,spy,c=[0,0,0],edgecolors='white',s=100,zorder=1000)
      ax.set_xlim([0,1])
      ax.set_ylim([0,1])
      ax.set_xlabel(r'$\phi_1$')
      ax.set_ylabel(r'$\phi_2$')
      plt.xticks(np.array([.1,.3,.5,.7,.9]))
      ax.set_xticklabels(np.array([20,40,60,80,100]))
      plt.yticks(np.array([.1,.3,.5,.7,.9]))
      ax.set_yticklabels(np.array([20,40,60,80,100]))
      fig.savefig("kriging"+str(i)+".pdf")

class MyMIRTK():
   def __init__(self,params,n_resolution,c_points,parin):
      self.params = params
      self.n_resolution = n_resolution
      self.c_points = c_points
      self.parin = parin
      
   def setFiles(self,files):
      """
      files = {'ref_im': ref_im_path, 'ref_vtk': ref_vtk_path,
               'mov_im': mov_im_path, 'mov_vtk': mov_vtk_path',
               'out_im': out_im_path, 'out_vtk': out_vtk_path,
               'dofs': dofs_path}
      """
      self.files = files
      
   def setLimits(self):
      """
      This function sets the boundaries of objectives 1 and 2 for normalization
      """
      self.editParin(np.array(np.zeros(self.n_resolution+len(self.params))))
      self.mirtkRegister()
      min_obj1,max_obj2 = self.scoreObjectives()
      
      self.editParin(np.array(np.ones(self.n_resolution+len(self.params))))
      self.mirtkRegister()
      max_obj1,min_obj2 = self.scoreObjectives()
      
      self.limits = [min_obj1,max_obj1,min_obj2,max_obj2]

   def setPCA(self,db,exp_var_ratio):
      face_db = np.array([np.ravel(db['images'][i]) for i in range(db['images'].shape[0])])
      mean_face = np.mean(face_db,0)
      centered_face_db = face_db - mean_face
      
      c0 = 0
      c1 = 100
      c = (c0+c1)/2
      found = False
      pca = PCA(n_components=c).fit(centered_face_db)
      
      while not found:
         if np.sum(pca.explained_variance_ratio_)>exp_var_ratio:
            c1 = c
            c = (c0+c1)/2
         else:
            c0 = c
            c = (c0+c1)/2
         pca = PCA(n_components=c).fit(centered_face_db)
         if c0 == c1 or c0 == c1-1:
            found = True
         print 'No. of components = '+str(c)+ \
             '\nExplained variance ratio = '+str(np.sum(pca.explained_variance_ratio_))
             
      pca = PCA(n_components=c+1).fit(centered_face_db)
      
      self.pca = pca
      self.mean_face = mean_face
   
   def mirtkRegister(self):
      """
      Subroutine for MIRTK command line (Linux)
      Error while loading shared libraries.\n \
      Maybe 'sudo ldconfig opt/mirtk/lib' will create a link to your project folder."
      """
      #Generate transformation file
      call(['/opt/mirtk/bin/./mirtk','register','-parin',self.parin,'-image',
      self.files['ref_im'],'-pset',self.files['ref_vtk'],'-image',self.files['mov_im'],'-pset',
      self.files['mov_vtk'],'-dofout',self.files['dofs']])
                
      #Apply transformation to image
      call(['/opt/mirtk/bin/./mirtk','transform-image',self.files['mov_im'],
      self.files['out_im'],'-dofin',self.files['dofs']])
      
      #Apply transformation to landmarks
      call(['/opt/mirtk/bin/./mirtk','transform-points',self.files['mov_vtk'],
      self.files['out_vtk'],'-source',self.files['mov_im'],'-target',self.files['ref_im'],
      '-dofin',self.files['dofs'],'-ascii','-invert'])
      
      with open(self.files['out_vtk'], 'rt') as fin:
         with open('temp', 'wt') as fout:
            fout.write(fin.read().replace(' 0 ', ' 0\n'))
      os.rename('temp', self.files['out_vtk'])
      call(['sed','-i','/^$/d',self.files['out_vtk']])
   
   def editParin(self,X):
      """
      Edit parin files according to values from array X
      """
      params_dic = {'imw': [23,"Image (dis-)similarity weight =",0,1],
                    'psdw':[24,"Point set distance weight     =",0,1],
                    'bew': [25,"Bending energy weight         =",0,.1]}
      aux = [82,85,88,91,94,97]
      with open(self.parin, 'rt') as fin:
         with open('temp', 'wt') as fout:
            f_edit = fin.readlines()
            f_edit[13] = "No. of resolution levels =" + str(self.n_resolution) + "\n" 
            if len(self.params) > 0:
               for i in range(len(self.params)):
                  f_edit[params_dic[self.params[i]][0]] = params_dic[self.params[i]][1] + \
                                             str(X[i]*params_dic[self.params[i]][3] + \
                                                       params_dic[self.params[i]][2]) + "\n"
            if self.c_points == True:
               for i in range(self.n_resolution):
                  f_edit[aux[i]] = "Control point spacing=" + \
                                           str(np.round(X[i+len(self.params)]*190)+10) + "\n"
            fout.write(''.join(map(str, f_edit)))
      os.rename('temp', self.parin)
   
   def scoreObjectives(self):
      """
      This function returns a registration score for given parameters
      
      OUTPUT
      obj1: euclidean distance between landmarks
      obj2: euclidean distance between principal components
      """
      ###Objective 1 - Landmarks Euclidean Distance
      with open(self.files['ref_vtk'], 'rt') as f_vtk1:
         vtk_list1 = f_vtk1.readlines()
      with open(self.files['out_vtk'], 'rt') as f_vtk2:
         vtk_list2 = f_vtk2.readlines()
      dist_array = np.array([])
      for i in range(5,len(vtk_list1)):
         landmark1 = np.asarray(vtk_list1[i].split(),dtype=np.float64)
         landmark2 = np.asarray(vtk_list2[i].split(),dtype=np.float64)
         dist = np.sqrt(np.sum((landmark1-landmark2)**2))
         dist_array = np.append(dist_array,dist)
         
      obj1 = np.sum(dist_array)
      
      ###Objective 2 - Face Components Euclidean Distance
      face1 = np.ravel(io.imread(self.files['mov_im']))
      face2 = np.ravel(io.imread(self.files['out_im']))
                           
      face1[face1==0] = self.mean_face[face1==0]
      pca_face1 = self.pca.transform((face1-self.mean_face).reshape(1, -1))
      
      face2[face2==0] = self.mean_face[face2==0]
      pca_face2 = self.pca.transform((face2-self.mean_face).reshape(1, -1))
      
      obj2 = np.sqrt(np.sum((pca_face1-pca_face2)**2))
      
      return obj1, obj2
      
   def costFun(self,X):
      try:
         X.shape[1]
      except:
         X = np.array([X])
 
      f = np.empty(X.shape[0])
      for i in range(X.shape[0]):
         self.editParin(X[i,:])
         self.mirtkRegister()
         norm_obj1 = 1/(self.limits[1]-self.limits[0])*(self.scoreObjectives()[0]-self.limits[1])+1
         norm_obj2 = 1/(self.limits[3]-self.limits[2])*(self.scoreObjectives()[1]-self.limits[3])+1
         f[i] = np.clip(norm_obj1,0,1)**2 + np.clip(norm_obj2,0,1)**2
         #print X[i,:]
         #print f[i]
      return f

   def krigeRegister(self,files,samples,numiter):
      self.setFiles(files)
      self.setLimits()
      
      if self.c_points == True:
         dim = self.n_resolution + len(self.params)
         X = lhs(dim, samples=samples, criterion='maximin')
      else:
         dim = len(self.params)
         X = lhs(dim, samples=samples, criterion='maximin')
         
      f = self.costFun
      print "Sampling space..."
      y = f(X)
      model = MyKriging(X, y, name='simple')
      print "Training kriging model..."
      model.train()
      for i in range(numiter):  
         print 'Infill iteration {0} of {1}....'.format(i + 1, numiter)
         newpoints = model.infill(1,method='ei')
         for point in newpoints:
            model.addPoint(point, f(point)[0])
         model.train()
      print "Predicting optimal solution..."
      self.kdata = model.kdata()
      del model
      position= np.where(self.kdata[2]==np.min(self.kdata[2]))
      self.solution = np.array([self.kdata[0][position[0][0],position[1][0]],
                                self.kdata[1][position[0][0],position[1][0]]])
      self.editParin(self.solution)
      print "Registering image..."
      self.mirtkRegister()
      print "Done!"