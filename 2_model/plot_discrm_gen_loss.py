
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def smooth_curve_gen(points, factor=0.6):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def smooth_curve_dis(points, factor=0.3):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

'''
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)


x = np.arange(0.2*np.pi, 0.02)
y = np.sin (x) + np.random.rand(len(x))
plt.plot (x,y,'r')
yhat = smooth_curve(y)
plt.plot(x, yhat, 'b')
plt.show()
'''
"""
Gen_loss = np.load('C:/Users/pegah/Desktop/Result/r_30_msi/pic/Gen_loss.npy')
s=np.argsort(Gen_loss)
p=[]
for i in range(len(s)):
    if s[i]%600==0:
        p.append(s[i])

Disc_loss_real = np.load('C:/Users/pegah/Desktop/Result/r_30_msi/pic/Disc_loss_real.npy')
m=np.argsort(Disc_loss_real)
k=[]
for i in range(len(m)):
    if m[i]%600==0:
        k.append(m[i])
"""
Gen = np.load('/Gen_loss.npy')
Disc_real = np.load('/Disc_loss_real.npy')
Disc_fake =  np.load('/Disc_loss_fake.npy')


Disc_loss_real = []
Disc_loss_fake = []
Gen_loss = []

for i in range(len(Gen)):
    if  i % 450 == 0:
        Gen_loss.append(Gen[i])
        Disc_loss_fake.append(Disc_real[i])
        Disc_loss_real.append(Disc_fake[i])
'''
import pandas as pd

df = pd.DataFrame(Gen_loss)
df.to_csv('C:/Users/pegah/Desktop/Gen_loss.csv',index=False)
d = pd.DataFrame(Disc_loss_fake)
d.to_csv('C:/Users/pegah/Desktop/Disc_loss_fake.csv',index=False)
dfh = pd.DataFrame(Disc_loss_real)
dfh.to_csv('C:/Users/pegah/Desktop/Disc_loss_real.csv',index=False)
'''
def demo(sty):
	epoch = range(0, len(Gen_loss))
	mpl.style.use(sty)
	fig, ax = plt.subplots(figsize=(6, 4))
	plt.plot(epoch, Disc_loss_real, linewidth=0.8, color='#9b0000', marker='.')
	plt.title('Discriminator loss real')
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.show()
	fig, ax = plt.subplots(figsize=(6, 4))
	plt.plot(epoch, smooth_curve_dis(Disc_loss_fake), linewidth=0.8, color='#1562ff' ,marker='.')
	plt.title('Discriminator loss fake')
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.show()
	fig, ax = plt.subplots(figsize=(6, 4))
	plt.plot(epoch, smooth_curve_gen(Gen_loss),linewidth=0.8, color='#8a3ac6' ,marker='.')
	plt.title('Generator loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()
    
demo('seaborn')


