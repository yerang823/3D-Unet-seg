#-*- coding:utf-8 -*-

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train import tversky_loss,dice_cost_0,dice_cost_1,dice_cost_01,dice_cost_1_loss
import glob


def preds_Thresh(preds_test,Thresh_value,dcm_cnt):
  
    preds_mask = np.ndarray((len(preds_test),dcm_cnt,128, 128, 2), dtype=np.float32)
  
    preds_mask0 = preds_test[:,:,:,:,0] > Thresh_value
    preds_mask1 = preds_test[:,:,:,:,1] > Thresh_value
  
    preds_mask[:,:,:,:,0] = preds_mask0*1
    preds_mask[:,:,:,:,1] = preds_mask1*1
  
    return preds_mask



def plot_roc_curve(fpr,tpr,auc):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr,tpr) 
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False positive rate') 
    plt.ylabel('True positive rate') 
    plt.legend(["AUC=%.3f"%auc])
    plt.show()    
    


#X = np.load('../data/resam/19_img.npy', allow_pickle=True)
#Y = np.load('../data/resam/19_ytrain.npy', allow_pickle=True)

li_ori=glob.glob('../rawdata/external/npy_ori_resam/*.npy')
li_mask=glob.glob('../rawdata/external/npy_mask_resam/*.npy')

X=[]
Y=[]
for i in range(len(li_ori)):
    X.append(np.load(li_ori[i]))
    Y.append(np.load(li_mask[i]))
    print(i,'/',len(li_ori))
X=np.array(X)
Y=np.array(Y)


train_dic, test_dic, train_label, test_label = train_test_split(X, Y, test_size=0.2)
test_dic, test_x, test_label, test_y = train_test_split(test_dic, test_label, test_size=0.5)

print(train_dic.shape, np.max(train_dic), np.min(train_dic))
print(train_label.shape, np.max(train_label), np.min(train_label))
print(test_dic.shape, np.max(test_dic), np.min(test_dic))
print(test_label.shape, np.max(test_label), np.min(test_label))
print(test_x.shape, np.max(test_x), np.min(test_x))
print(test_y.shape, np.max(test_y), np.min(test_y))


train_dic = train_dic.astype(np.float32)
test_dic = test_dic.astype(np.float32)
train_label = train_label.astype(np.float32)
test_label = test_label.astype(np.float32)
test_x = test_x.astype(np.float32)
test_y = test_y.astype(np.float32)

model = load_model('../result/model/3Dunet_best_210113.h5',
                   custom_objects = {'tversky_loss': tversky_loss,
                                     'dice_cost_1_loss': dice_cost_1_loss,
                                     'dice_cost_0': dice_cost_0,
                                    'dice_cost_1': dice_cost_1,
                                    'dice_cost_01': dice_cost_01})
                                    
                                    
preds_test = model.predict(test_x, verbose=1)

print(test_x.shape)
print(test_y.shape)

Dcm_Cnt=X.shape[0] ### 
preds_mask = preds_Thresh(preds_test,0.1, Dcm_Cnt)

toto=0
for i in range(len(test_x)):
    plt.figure(figsize=(15, 15))
    plt.subplot(1,4,1)
    plt.imshow(test_x[toto,i,:,:,0],cmap='gray')
    plt.title('CT image')
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(test_y[toto,i,:,:,1],cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.imshow(preds_mask[toto,i,:,:,1],cmap='gray',)
    plt.title('Prediction Result')
    plt.axis('off')
    plt.subplot(1,4,4)
    plt.imshow(test_x[toto,i,:,:,0],cmap='gray')
    plt.imshow(preds_mask[toto,i,:,:,1],cmap='Reds',alpha=0.25)
    plt.title('CT + Result')
    plt.axis('off')
    #plt.savefig('test.png')


seg1 = preds_mask[:,:,:,1].flatten()
gt1 = test_y[:,:,:,1].flatten()

print(seg1.shape)
print(gt1.shape, "\n")

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(gt1, seg1)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(gt1, seg1)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(gt1, seg1)
print('Recall: %f' % recall)

# Dice similarity coefficient
k = 1
dice = np.sum(seg1[gt1==k])*2.0 / (np.sum(seg1) + np.sum(gt1))
print('Dice similarity score: {}'.format(dice))


prob = preds_test[:,:,:,1]
gt = test_y[:,:,:,1]

# plot 1
precision, recall, thresholds = precision_recall_curve(gt.flatten().astype(np.bool), prob.flatten())
ap = average_precision_score(gt.flatten().astype(np.bool), prob.flatten())
plot_roc_curve(precision,recall,ap)

# plot 2
fpr, tpr, _ = roc_curve(gt.flatten().astype(np.bool), prob.flatten())
auc = roc_auc_score(gt.flatten().astype(np.bool), prob.flatten())
plot_roc_curve(fpr,tpr,auc)


'''
preds_mask = preds_Thresh(preds_test,0.2)

pbar = tqdm_notebook(total=len(preds_mask))

for toto in range(len(preds_mask)):
  
    plt.figure(figsize=(15, 15))
    plt.subplot(1,4,1)
    plt.imshow(test_x[toto,:,:,0],cmap='gray')
    plt.title('Original CT image')
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(test_y[toto,:,:,1],cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.imshow(preds_mask[toto,:,:,1],cmap='gray')
    plt.title('Prediction Result')
    plt.axis('off')
    plt.subplot(1,4,4)
    plt.imshow(test_x[toto,:,:,0],cmap='gray')
    plt.imshow(preds_mask[toto,:,:,1],cmap='Reds',alpha=0.25)
    plt.title('CT + Result')
    plt.axis('off')
  
    resfilename = '/home/ubuntu/disk1/pancreas_segmentation/result[2020.02.06]/' + str(toto+1).zfill(5) + '.png'
  
    #if not os.path.exists(resfoldername):
    #    os.makedirs(resfoldername)
        
    plt.savefig(resfilename, bbox_inches='tight', pad_inches=1)
  
    plt.close()
  
    pbar.update(1)

pbar.close()
'''