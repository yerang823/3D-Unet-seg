#-*- coding:utf-8-*-
import cv2
import numpy as np
import glob
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
from tqdm import tnrange, tqdm_notebook
import pandas as pd

def resample_img(itk_image, slice_thickness, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
        
    out_size = [
       int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))), # x,y axis 수정 O
       int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
       int(np.round(original_size[2] * (slice_thickness / out_spacing[2]))),]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
        
    return resample.Execute(itk_image)



def eliminate_blackCnts(image):
    # ============================ 쓸모없는 슬라이스 걸러내기 ==============================
    ssl_slice = []

    for i in range(len(image)):
        if image[i,:,:].min() == image[i,:,:].max() and image[i,:,:].min() == 170 and image[i,:,:].max() == 170:
            ssl_slice.append(i)

    front_max=0
    back_min=9999
    for i in range(len(ssl_slice)):
        #앞에서 제일 큰것
        if ssl_slice[i]< (len(image)/2) and ssl_slice[i]>front_max :
            front_max=ssl_slice[i]
        #뒤에서 제일 작은 것
        elif ssl_slice[i] > (len(image)/2) and ssl_slice[i]<back_min :
            back_min=ssl_slice[i]
    
    print("len(ssl_slice)=",len(ssl_slice))  
    
    return front_max,back_min
  
  


def getSliceThickness(Dcm_dir):
    #==================== slice thickness 가져오기 ========================================
    reader = sitk.ImageFileReader()
    reader.SetFileName(Dcm_dir)
    reader.ReadImageInformation()
    thickness=float(reader.GetMetaData('0018|0050'))
    # ======================================================================================  
  
    return thickness



def cutCntsFromOriImg(img,mask,front_max,back_min):
    if front_max!=0 and back_min==9999: # 앞에만있는경우
        img=img[:,:,front_max:] # x,y,z
        mask=mask[:,:,front_max:] # x,y,z
    elif front_max==0 and back_min!=9999: # 뒤에만있는경우
        img=img[:,:,:back_min] # x,y,z
        mask=mask[:,:,:back_min] # x,y,z
    
    elif front_max!=0 and back_min!=9999: # 둘다 있는 경우
        img=img[:,:,front_max:back_min] # x,y,z
        mask=mask[:,:,front_max:back_min] # x,y,z
    else: # 둘다 없는 경우
        img=img[:,:,:] # x,y,z
        mask=mask[:,:,:] # x,y,z
    
    return img,mask


def makeYtrain(Mask_vol):
    print("makeYtrain..........................")
    patients=len(Mask_vol.shape)
    z=Mask_vol[0].shape[0]
    y=Mask_vol[0].shape[1]
    x=Mask_vol[0].shape[2]
    
    Y_train = np.ndarray((patients,z,y,x,2), dtype=np.float32)
    for i in range(patients):
        
        mask=Mask_vol[i]
        
        Y_train0 = mask < .5   # 참, 거짓 (bool)
        Y_train1 = mask > .5  # label
    
        Y_train0=np.reshape(Y_train0,(1,z,y,x,1))
        Y_train1=np.reshape(Y_train1,(1,z,y,x,1))
    
        Y_train[ i, :,:,:,0] = Y_train0[:,:,:,:,0]*1.0 # bool -> 숫자로
        Y_train[ i, :,:,:,1] = Y_train1[:,:,:,:,0]*1.0
    
        print(i,'/',patients)
        print(Y_train0.shape, np.max(Y_train0), np.min(Y_train0))
        print(Y_train1.shape, np.max(Y_train1), np.min(Y_train1))
        print(Y_train.shape, np.max(Y_train), np.min(Y_train))
    return Y_train




def resample_main(nodule_num):
    dcm_folders=r'/root/gcubme4/Workspace/YR_Park/yerang_54/10_noduleInsertion_KHJ/06_CT-GAN/exp3_noduleDataAll/data/scans'
    
    folder_names=os.listdir(dcm_folders) #300개 폴더
    folder_names.sort()
    
    #ready_resam=os.listdir('../data/15/resam_tmp/img/') #300개 폴더
    ready_resam=os.listdir('../rawdata/manual/npy_15/')
    ready_resam.sort()
    print("len(ready_resam)=",len(ready_resam))
    
    print('Npy loading......')
    vol_stack=np.load('../data/manual/%d/2_CTs_seg.npy'%nodule_num,allow_pickle=True)
    print('vol loaded')
    label_stack=np.load('../data/manual/%d/2_labels.npy'%nodule_num,allow_pickle=True)
    print('Loaded')
    
    img_vol=[]
    mask_vol=[]
    csv=[]
    for num in range(len(folder_names)):
        csv_tmp=[] # 슬라이스 제거후 개수 csv로 저장하기 위함
        dcm_dir = sorted(glob.glob(dcm_folders+'/%s/*.dcm'%folder_names[num]))
    
        origin_dcm = sitk.ReadImage(dcm_dir)
        origin_spacing = origin_dcm.GetSpacing()
        
        thickness = getSliceThickness(dcm_dir[0])
        
        
        print(folder_names[num], origin_spacing, thickness)
        
        #npy_name=folder_names[num].zfill(3)
        
        image=vol_stack[num]
        label=label_stack[num]
        print("ORIGINAL")
        print("image=",image.shape)
        print("label=",label.shape)
        csv_tmp.append(image.shape[0]) ######information
    #     print(image.dtype)
    #     print(label.dtype)
    #     print(np.max(image))
    #     print(np.min(label))
    
        
        front_max,back_min = eliminate_blackCnts(image)   
    
        
        # ======================================================================================
    
        img = sitk.GetImageFromArray(image[:,:,:], isVector=False) 
        mask = sitk.GetImageFromArray(label[:,:,:], isVector=False)
    
        img.CopyInformation(origin_dcm)
        mask.CopyInformation(origin_dcm)
            
        
        # cut off slices based on front_max, back_min
        img,mask = cutCntsFromOriImg(img,mask,front_max,back_min)
        
       
        print("AFTER REMOVING SLICES")
        print(img.GetSize())
        print(mask.GetSize())
        csv_tmp.append(img.GetSize()[2]) ######information
        
    
        x_y_spacing = origin_spacing[0] # 원본 x,y spacing
        z_spacing = img.GetSize()[2]*thickness / 128 ## slice number to get after resample
    
        
        filtered_image = resample_img(img, thickness, out_spacing=[x_y_spacing*4, x_y_spacing*4, z_spacing], is_label=False)
        filtered_mask = resample_img(mask, thickness, out_spacing=[x_y_spacing*4, x_y_spacing*4, z_spacing], is_label=True)
        
        print(origin_spacing[:2],thickness, "---->", filtered_image.GetSpacing())
        
        #normalize
        img = sitk.GetArrayFromImage(filtered_image) / 255.0
        mask = sitk.GetArrayFromImage(filtered_mask) / 255.0
        
        # mask 0,1로 thresholding 해주기
        mask[mask<0.5] =0
        mask[mask>0.5] =1
        
        img = np.expand_dims(img, axis=4)
        mask = np.expand_dims(mask, axis=4)
        
        print("AFTER RESAMPLE")
        print(img.shape)
        print(mask.shape)
        
        #np.save('../data/%d/resam_tmp/img/%s.npy'%(nodule_num,folder_names[num]), img)
        #np.save('../data/%d/resam_tmp/mask/%s.npy'%(nodule_num,folder_names[num]), mask)
        np.save('../rawdata/manual/resam_%d/img/%s.npy'%(nodule_num,folder_names[num]), img)
        np.save('../rawdata/manual/resam_%d/mask/%s.npy'%(nodule_num,folder_names[num]), mask)
        
        img_vol.append(img)
        mask_vol.append(mask)
        csv.append(csv_tmp)
    
    #     print(img.dtype)
    #     print(mask.dtype)
    #     print(np.max(img))
    #     print(np.min(mask))
        
        print("================================================")
    
    
    img_vol_arr=np.array(img_vol)
    mask_vol_arr=np.array(mask_vol)
    np.save('../data/manual/%d/241_CTs_resam.npy'%nodule_num,img_vol_arr)
    np.save('../data/manual/%d/241_labels_resam.npy'%nodule_num,mask_vol_arr)
    Y_train=makeYtrain(mask_vol_arr)
    np.save('../data/manual/%d/241_ytrain.npy'%nodule_num, Y_train)
    
    csv_df=pd.DataFrame(csv)
    csv_df.to_csv('../data/manual/%d/241_resampleInfo.csv'%nodule_num,'w',header=None,index=False)
    

#print("15=========================")
#resample_main(15)
#print("30=========================")
#resample_main(30)
print("45=========================")
resample_main(45)