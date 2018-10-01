import os
import glob
import numpy as np
from sanrr import SANRR, create_fei_db, save_files

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MIRTK_DIR = os.path.join(ROOT_DIR,'mirtk_folder')

fei_db = create_fei_db()
save_files(fei_db, os.path.join(MIRTK_DIR, 'im/'))

configuration_file = os.path.join(MIRTK_DIR, 'register-2d-face-landmarks.cfg')
solver = SANRR([], 2, True, configuration_file)

solver.set_PCA(fei_db['images'], .8, [0,100])

files = {'ref_im': os.path.join(MIRTK_DIR, 'im/mean/mean.pgm'),
         'ref_vtk': os.path.join(MIRTK_DIR, 'im/mean/mean.vtk'),
         'dofs': os.path.join(MIRTK_DIR, 'transformed/b__a.dof.gz')}

for name in glob.glob(MIRTK_DIR + '/im/*.pgm'):
    name = os.path.basename(name)[:-4]
    files['mov_im'] = os.path.join(MIRTK_DIR, 'im/' + name + '.pgm')
    files['mov_vtk'] = os.path.join(MIRTK_DIR, 'im/' + name + '.vtk')
    files['out_im'] = os.path.join(MIRTK_DIR, 'transformed/' + name + '.pgm')
    files['out_vtk'] = os.path.join(MIRTK_DIR, 'transformed/' + name + '.vtk')
    
    np.random.seed(0)
    solver.krige_register(files, 11, 5)