
import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import distance

ED = np.zeros((500,1), dtype='float32')
dst = np.zeros((500,1), dtype='float32')

for i in range(500):
    a = mpimg.imread('/Ground_truth/Bg/%d.tiff' % (i+1))
    b = mpimg.imread('/stst/Bg/%d.tiff' % (i+1))
    
    c = np.array((a-np.min(a))/(np.max(a)-np.min(a)))
    d = np.array((b-np.min(b))/(np.max(b)-np.min(b)))
    
    m = np.ravel(c)
    n = np.ravel(d)
    
    
    
    dst[i][0] = distance.euclidean(m, n)
    ED[i][0] = np.linalg.norm(m-n)
    
    if (i % 100) == 0:
        print ('step: ', i)
        
avg_ED = np.average(ED, axis = 0)
std_ED = np.std(ED, axis = 0)

avg_dst = np.average(dst, axis = 0)
std_dst = np.std(dst, axis = 0)