import numpy as np
import pandas as pd
import os,glob
import cv2


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


NUM=15
root='../rawdata/%d/raw'%NUM
li=os.listdir(root)

raw_list=[]
for k in range(len(li)):
	if li[k].split('.')[-1]=='raw':
		raw_list.append(li[k])

		
npy_li=[]
for i in range(len(raw_list)):
	folder_name=int(raw_list[i].split('.')[0])
	
	f=open(root+raw_list[i], 'rb')
	npy=np.fromfile(f,np.int16)
	x=512
	y=512
	z=int(len(npy)/(512*512))
	data_array = npy.reshape((z,y,x))
	data_transform = lumTrans(data_array)
	npy_li.append(data_transform)
	print('data_transform.shape=',data_transform.shape,
			i,'/',len(raw_list))
	middle_cnt=round(data_transform.shape[0]/2)
	cv2.imwrite('../rawdata/%d/center_img_tocheck/%s.png'%(NUM,str(folder_name)), data_transform[middle_cnt])


npy_stack=np.array(npy_li)
print('npy_stack.shape=',npy_stack.shape)
np.save('../data/%d/npy/300_CTs_210120.npy'%NUM,npy_stack)