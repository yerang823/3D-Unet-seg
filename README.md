# =========================================================
# 	RANDOM GAN NODULE
# =========================================================

# 1. Preprocess
    python raw2npy_stack.py  
  - individual raw -> stack npy (ct) 
  - shape : patients, cnts(different), x, y
  
    python csv2maskNpy_stack.py
  - individual csv(45 of random seed list) -> stack npy (label)
  - shape : patients, cnts(different), x, y, 1

    python resample.py
  - remove black slices and resample to 150,128,128,1 (ct, label)
  - at the end, label also saved as y_train (shape=[:,:,:,:,2])
  - save 300_ct_resam.py, 300_label_resam.py, 300_ytrain.py

  - check that resampled cts and labels are well matched using 'visualization.ipynb'.
  
# 2. train.py

# 3. test.py
  - test and evaluate
  
 
 
 