### Result1 : Randomly inserted GAN nodule detection
(https://user-images.githubusercontent.com/97432613/156339841-adb277ec-ac5f-4286-8d86-506930150ae0.png)

### Result2 : Manually inserted additional original nodule detection
(https://user-images.githubusercontent.com/97432613/156339861-eabeb960-11e5-4094-b428-6acb5eb5b003.png)

# 1. Data preprocessing

    python raw2npy_stack.py  

  - individual raw -> stack npy (ct) 
  - shape : patients, cnts(different), x, y
```python
  python csv2maskNpy_stack.py
```
  - individual csv(45 of random seed list) -> stack npy (label)
  - shape : patients, cnts(different), x, y, 1
```python
  python resample.py
```
  - remove black slices and resample to 150,128,128,1 (ct, label)
  - at the end, label also saved as y_train (shape=[:,:,:,:,2])
  - save 300_ct_resam.py, 300_label_resam.py, 300_ytrain.py

  - check that resampled cts and labels are well matched using 'visualization.ipynb'.
  
# 2. train

    python train.py


# 3. test

    python test.py

  - test and evaluate
  
 
 
 