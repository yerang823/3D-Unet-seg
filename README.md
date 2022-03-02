[[Full Paper Here](https://arxiv.org/pdf/1804.03999.pdf)]

## Result1 : Randomly inserted GAN nodule detection
<p align="center">
    <img src="https://user-images.githubusercontent.com/97432613/156339841-adb277ec-ac5f-4286-8d86-506930150ae0.png"  width="80%" height="80%"/>
</p>

## Result2 : Manually inserted additional original nodule detection
<p align="center">
    <img src="https://user-images.githubusercontent.com/97432613/156339861-eabeb960-11e5-4094-b428-6acb5eb5b003.png"  width="80%" height="80%"/>
</p>

## 3D segmentation workflow
<p align="center">
    <img src="https://user-images.githubusercontent.com/97432613/156349405-42fc19ca-de28-4a62-abc7-9fff7ce38a23.png"  width="80%" height="80%"/>
</p>

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
  
# Final result
![result_1](https://user-images.githubusercontent.com/97432613/156354252-77b71722-c62e-4b25-be1f-f96871191555.png)

![result_2](https://user-images.githubusercontent.com/97432613/156354255-2b789721-8428-41d7-a364-37871ddb5664.png)

 <p align="center">
    <img src="https://user-images.githubusercontent.com/97432613/156354248-31937a9a-5b20-49b0-a144-2df921b15333.png"  width="50%" height="50%"/>
</p>
 
