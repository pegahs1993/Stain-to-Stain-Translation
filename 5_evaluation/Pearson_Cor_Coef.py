
import numpy as np
from scipy.stats import pearsonr
from PIL import Image
import numpy as geek 

pcc_sp = np.zeros((500,3), dtype='float32')


for i in range(500):
    orginal = Image.open('/Ground_truth/%d.tiff'% (i+1))
    generated = Image.open('/stst/%d.tiff'% (i+1))
    
    org = np.array(orginal)
    gen = np.array(generated)
    
    O = org.ravel()
    G = gen.ravel()
    
    pcc ,p = pearsonr(O,G)
    
    pcc_sp[i][0] = pcc
    
    if (i % 100) == 0:
        print ('step: ', i)


value_sort_pcc = np.sort(pcc_sp, axis=0)
index_sort_pcc = geek.argsort(pcc_sp, axis = 0)   

avg = np.average(pcc_sp, axis = 0)
std = np.std(pcc_sp, axis = 0)
      
for i in range(500):
    pcc_sp[i][1] = value_sort_pcc[i][0]
    pcc_sp[i][2] = index_sort_pcc[i][0]+1


#np.save('/pcc_sp.npy', pcc_sp)        
    















