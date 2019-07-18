# concrete
This is a civil engineering application for understanding machine learning.
In order to run a program in this folder, you must install python using miniconda:
<pre>
https://docs.conda.io/en/latest/miniconda.html
</pre>
It is recommended to install Python2.7.X instead of Python3.X. However, Python2.7.X will retire on Jan.1 2020. The programs were tested using Python2.7.X and Python3.7.X.
There are six programs including concrete_ext.py, concrete_rf.py, concrete_rf2.py, concrete_rf3.py, concrete_rf33.py, concrete_rf4.py.
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
</pre>

