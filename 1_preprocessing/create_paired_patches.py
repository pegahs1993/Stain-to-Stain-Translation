
# Preparing dataset with paired patches

from __future__ import print_function
import os

from PIL import Image


result = Image.new("RGB", (512, 256))

for i in range(12720):
    # Create grayscale patches
    img_gray = Image.open('/H/%d.tiff' % (i+1)).convert('LA')
    img_gray.save('/HG/%d.tiff' % (i+1))
    
    files = [
            ('/H/%d.tiff' % (i+1)),
            ('/HG/%d.tiff' % (i+1))
            ]
    for index, file in enumerate(files):
        path = os.path.expanduser(file)
        img = Image.open(path)
        img.thumbnail((256, 256), Image.ANTIALIAS)
        y = index // 2 * 256
        x = index % 2 * 256
        w, h = img.size
        print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h))
        result.save(os.path.expanduser('/H_HG/%d.tiff' % (i+1)))
        

