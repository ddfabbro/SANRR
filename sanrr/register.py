import os
import numpy as np
from skimage import io
from subprocess import call
from pyDOE import lhs
from metamodel import MyKriging

#Import git submodule
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "expvar-pca"))
from expvarpca import PCA
 
class SANRR():
    def __init__(self,params,n_resolution,c_points,parin):
        self.params = params
        self.n_resolution = n_resolution
        self.c_points = c_points
        self.parin = parin
      
    def set_files(self,files):
        """
        files = {
            'ref_im': ref_im_path, 
            'ref_vtk': ref_vtk_path,
            'mov_im': mov_im_path, 
            'mov_vtk': mov_vtk_path',
            'out_im': out_im_path, 
            'out_vtk': out_vtk_path,
            'dofs': dofs_path,
        }
        """
        self.files = files
      
    def set_limits(self):
        """
        Sets the boundaries of objectives 1 and 2 for normalization
        """
        self.edit_parin(np.array(np.zeros(self.n_resolution+len(self.params))))
        self.mirtk_register()
        self.min_obj1 = self.score_landmarks()
        self.max_obj2 = self.score_PCA()
        
        self.edit_parin(np.array(np.ones(self.n_resolution+len(self.params))))
        self.mirtk_register()
        self.max_obj1 = self.score_landmarks()
        self.min_obj2 = self.score_PCA()
        
    def set_PCA(self, db, target_exp_var, bounds):
        face_db = np.array([np.ravel(db[i]) for i in range(db.shape[0])])
        mean_face = np.mean(face_db,0)
        centered_face_db = face_db - mean_face
        
        pca = PCA(whiten=True,svd_solver='randomized')
        pca.fit_exp_var(centered_face_db, target_exp_var, bounds)
        
        self.pca = pca
        self.mean_face = mean_face
   
    def mirtk_register(self):
        """
        Subroutine for MIRTK command line (Docker)
        """
        #Generate transformation file
        call(['docker','exec','mirtk','/usr/local/bin/./mirtk',
              'register',
              '-parin', self.parin,
              '-image', self.files['ref_im'],
              '-pset', self.files['ref_vtk'],
              '-image', self.files['mov_im'],
              '-pset', self.files['mov_vtk'],
              '-dofout', self.files['dofs']])
                
        #Apply transformation to image
        call(['docker','exec','mirtk','/usr/local/bin/./mirtk',
              'transform-image',
              self.files['mov_im'], 
              self.files['out_im'], 
              '-dofin', self.files['dofs']])
      
        #Apply transformation to landmarks
        call(['docker','exec','mirtk','/usr/local/bin/./mirtk',
              'transform-points',
              self.files['mov_vtk'],
              self.files['out_vtk'],
              '-source', self.files['mov_im'],
              '-target',self.files['ref_im'],
              '-dofin',self.files['dofs'],
              '-ascii','-invert'])
      
        with open(self.files['out_vtk'], 'rt') as fin:
            with open('temp', 'wt') as fout:
                fout.write(fin.read().replace(' 0 ', ' 0\n'))
        os.rename('temp', self.files['out_vtk'])
        call(['sed','-i','/^$/d',self.files['out_vtk']])
   
    def edit_parin(self, X):
        """
        Edit parin files according to values from array X
        """
        params_dic = {
            'imw': [23,"Image (dis-)similarity weight =",0,1],
            'psdw':[24,"Point set distance weight     =",0,1],
            'bew': [25,"Bending energy weight         =",0,.1],
        }
        
        aux = [82,85,88,91,94,97]
        with open(self.parin, 'rt') as fin:
            with open('temp', 'wt') as fout:
                f_edit = fin.readlines()
                f_edit[13] = "No. of resolution levels =" + \
                                    str(self.n_resolution) + "\n" 
                                    
                if len(self.params) > 0:
                    for i in range(len(self.params)):
                        f_edit[params_dic[self.params[i]][0]] = \
                            params_dic[self.params[i]][1] + \
                            str(X[i]*params_dic[self.params[i]][3] + \
                            params_dic[self.params[i]][2]) + "\n"
                                
                if self.c_points == True:
                    for i in range(self.n_resolution):
                        f_edit[aux[i]] = "Control point spacing=" + \
                            str(np.round(X[i+len(self.params)]*190)+10) + "\n"
                fout.write(''.join(map(str, f_edit)))
        os.rename('temp', self.parin)
   
    def score_landmarks(self):
        """
        Objective 1 - Landmarks Euclidean Distance
        """
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
         
        return np.sum(dist_array)
    
    def score_PCA(self):
        """
        Objective 2 - Face Components Euclidean Distance
        """
        face1 = np.ravel(io.imread(self.files['mov_im']))
        face2 = np.ravel(io.imread(self.files['out_im']))
                           
        face1[face1==0] = self.mean_face[face1==0]
        pca_face1 = self.pca.transform((face1-self.mean_face).reshape(1, -1))
      
        face2[face2==0] = self.mean_face[face2==0]
        pca_face2 = self.pca.transform((face2-self.mean_face).reshape(1, -1))
      
        return np.sqrt(np.sum((pca_face1-pca_face2)**2))
      
    def cost_fun(self, X):
        try:
            X.shape[1]
        except:
            X = np.array([X])
 
        f = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            self.edit_parin(X[i,:])
            self.mirtk_register()
            
            obj1 = self.score_landmarks()
            obj2 = self.score_PCA()
            norm_obj1 = (obj1 - self.max_obj1)/(self.max_obj1 - self.min_obj1)+1
            norm_obj2 = (obj2 - self.max_obj2)/(self.max_obj2 - self.min_obj2)+1
            
            f[i] = np.clip(norm_obj1,0,1) + np.clip(norm_obj2,0,1)

        return f

    def krige_register(self, files, samples, numiter):
        self.set_files(files)
        self.set_limits()
      
        if self.c_points == True:
            dim = self.n_resolution + len(self.params)
            X = lhs(dim, samples=samples, criterion='maximin')
        else:
            dim = len(self.params)
            X = lhs(dim, samples=samples, criterion='maximin')
         
        f = self.cost_fun
        print("Sampling space...")
        y = f(X)
        model = MyKriging(X, y, name='simple')
        print("Training kriging model...")
        model.train()
        for i in range(numiter):  
            print('Infill iteration {0} of {1}....'.format(i + 1, numiter))
            newpoints = model.infill(1,method='ei')
            for point in newpoints:
                model.addPoint(point, f(point)[0])
            model.train()
        print("Predicting optimal solution...")
        self.kdata = model.kdata()
        del model
        position= np.where(self.kdata[2]==np.min(self.kdata[2]))
        self.solution = np.array([self.kdata[0][position[0][0],position[1][0]],
                                  self.kdata[1][position[0][0],position[1][0]]])
        self.edit_parin(self.solution)
        print("Registering image...")
        self.mirtk_register()
        print("Done!")
        