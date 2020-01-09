
# Create test and train set with random patches

import random
from PIL import Image

list=[]       
          

list = random.sample(range(12720), 12720)

k = 0
for i in range(12720):      
    j = list[i]
    img = Image.open('/H_HG/%d.tiff' % j)
    if i < 2999:
        img.save('/train/%d.tiff' % (i+1))
    else:
        img.save('/test/%d.tiff' % (k+1))
        k = k +1
