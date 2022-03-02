#-*-coding:utf-8-*-

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

 
def MakeMasks(csv_li, nodule_num):
    print("nodule_num ===============", nodule_num)
    
    # reversed list 
    f=open('../rawdata/manual/reversed_dcm_list.txt','r')
    reverse_li=f.readlines()
    f.close()
    idx=[]
    for i in range(len(reverse_li)):
      idx.append(reverse_li[i].split(',')[0])
    
    # make mask
    npy_vol=[]
    for i in range(len(csv_li)): 
        npy=np.zeros((cnt_li[i],512,512,1),dtype='uint8')
        
        if i in idx:
            print('i, cnt_li[i] =',i, cnt_li[i])
            for j in range(nodule_num): 
                x=csv_li[i][j][0]  
                y=csv_li[i][j][1]  
                z=cnt_li[i] - csv_li[i][j][2]  ###########
                
                npy[z-8:z+8,y-10:y+10,x-10:x+10,0]=255 # 20x20x16
        else:
            for j in range(nodule_num): 
                x=csv_li[i][j][0]  
                y=csv_li[i][j][1]  
                z=csv_li[i][j][2]  
                
                npy[z-8:z+8,y-10:y+10,x-10:x+10,0]=255 # 20x20x16
        
        npy_vol.append(npy)
    
    npy_vol_arr=np.array(npy_vol)
    print("npy_vol_arr.shape=",npy_vol_arr.shape)
    np.save('../data/manualgan/%d/241_labels.npy'%nodule_num, npy_vol_arr)



li=os.listdir('../rawdata/manualgan/manualSeedCsv')
li.sort()

csv_li=[]
for i in range(len(li)):
    if li[i].split('.')[-1]=='csv':
        csv=pd.read_csv('../rawdata/manualgan/manualSeedCsv/'+li[i],header=None)
        randomli=np.array(csv)
        csv_li.append(randomli)
        print(li[i])

        
# 각 dcm cnt 알아야 mask 만들수 있음
print("loading...")
ct_arr=np.load('../data/manual/15/241_labels.npy', allow_pickle=True)
print("loaded")


cnt_li=[]
for i in range(len(ct_arr)):
    cnt_li.append(ct_arr[i].shape[0])
    
    
MakeMasks(csv_li=csv_li, nodule_num=15)
MakeMasks(csv_li=csv_li, nodule_num=30)
MakeMasks(csv_li=csv_li, nodule_num=45)