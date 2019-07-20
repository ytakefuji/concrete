# concrete
There are two civil engineering applications for understanding machine learning: predicting concrete strength and detecting concrete cracks. MNIST dataset is introduced for image machine learning. SDNET2018 dataset is used for detecting concrete cracks.
In order to run a program in this folder, you must install python using miniconda:
<pre>
https://docs.conda.io/en/latest/miniconda.html
</pre>
It is recommended to install Python2.7.X instead of Python3.X. However, Python2.7.X will retire on Jan.1 2020. The programs were tested using Python2.7.X and Python3.7.X.
There are six programs including concrete_ext.py, concrete_rf.py, concrete_rf2.py, concrete_rf32.py, concrete_rf33.py, concrete_rf4.py.
For image machine learning using MNIST dataset, keras_cnn.py and mnist_torch.py are introduced. For detecting cracks using SDNET2018 dataset, ext_crack.py, rf_crack.py, and lgbm_crack.py are introduced. In order to convert images of SDNET2018 dataset into csv, there are two programs for converting images into csv file: createcsvCD.py for images in CD folder and createUD.py for images in UD folder respectively.
<pre>
concrete.csv: csv file of the date
Concrete_Data.xls: data from UCI machine learning
concrete_ext.py: extratrees algorithm
concrete_rf.py: randomforest algorithm
concrete_rf2.py: a single input to trained machine (error message will be generated)
concrete_rf2C.py: a single input to trained machine (error is corrected)
concrete_rf32.py: decision tree in Python2.X
concrete_rf33.py: decision tree in Python3.X
concrete_rf4.py: cross-validation
concrete_stack.py: stacking using rf and ext
keras_cnn.py: Deep Learning using Keras (reproducibility problem with multiple GPUs)
mnist_torch.py: Deep Learning using Pytorch (no reproducibility problem)
createcsvCD.py: Converting images in CD folder into cd_gray.csv file
createcsvUD.py: Converting images in UD folder into ud_gray.csv file
ext_crack.py: extratrees for detecting cracks
rf_crack.py: randomforest for detecting cracks
lgbm_crack.py: LightGBM for detecting cracks
</pre>

