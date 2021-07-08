## <center> Accuracy calculating of JIP models </center>

This Notebook calculates the mean accuracy of the train, validation and test results of models trained using JIP. Further the mean accuracies of the In Distribution and Out Of Distribution tests will be calculated.
Before executing this Notebook, be sure to have trained all 6 artefact models using the provided code in three steps:
    
1. Preprocess all datasets (train and test) using the following command:
```bash
python JIP.py --mode preprocess --device <cuda_id> --datatype train
```  
and   
```bash
python JIP.py --mode preprocess --device <cuda_id> --datatype test
```
2. Train all 6 models using the following command:
```bash
python JIP.py --mode train --device <cuda_id> --datatype train --noise_type <noise_model>
```
3. Perform the testing as follows:
```bash
python JIP.py --mode testIOOD --device <cuda_id> --datatype test --noise_type <noise_model>
```


Once this is finished, everything is set up to run the Notebook.

#### Import necessary libraries


```python
import os
import numpy as np
```

#### Set necessary directories
Specify the train_base and test_base directory. These are just the full paths to the JIP folder train_dirs and test_dirs output, for instance: `../JIP/train_dirs/output` and `../JIP/test_dirs/output`.


```python
# Set the base path to JIP/train_dirs/output folder
train_base = '<path>/JIP/train_dirs/output/'
# Set the base path to JIP/test_dirs/output folder
test_base = '<path>/JIP/test_dirs/output/'
```

#### Calculate the accuracies and print them


```python
# Load data for each artefact and calculate mean accuracy
artefacts = ['blur', 'ghosting', 'motion', 'noise', 'resolution', 'spike']

for artefact in artefacts:
    print('\nModel {}:'.format(artefact))
    train = np.load(os.path.join(train_base, artefact, 'results/accuracy_train.npy'))
    val = np.load(os.path.join(train_base, artefact, 'results/accuracy_validation.npy'))
    test = np.load(os.path.join(train_base, artefact, 'results/accuracy_test.npy'))
    test_ID = np.load(os.path.join(test_base, artefact, 'testID_results/accuracy_test.npy'))
    test_OOD = np.load(os.path.join(test_base, artefact, 'testOOD_results/accuracy_test.npy'))
    
    # Calculate train accuracy
    train_acc = 0
    for i in range(len(train)):
        train_acc += train[i][1]
    print('\tTrain acccuracy: %.2f' %(train_acc/len(train)) + '%')
    
    # Calculate validation accuracy
    val_acc = 0
    for i in range(len(val)):
        val_acc += val[i][1]
    print('\tValidation acccuracy: %.2f' %(val_acc/len(val)) + '%')
    
    # Calculate test accuracy
    test_acc = 0
    for i in range(len(test)):
        test_acc += test[i][1]
    print('\tTest (Dataloader) acccuracy: %.2f' %(test_acc/len(test)) + '%')
    
    # Calculate test_ID accuracy
    test_acc = 0
    for i in range(len(test_ID)):
        test_acc += test_ID[i][1]
    print('\tTest (In Distribution) acccuracy: %.2f' %(test_acc/len(test_ID)) + '%')
    
    # Calculate test_OOD accuracy
    test_acc = 0
    for i in range(len(test_OOD)):
        test_acc += test_OOD[i][1]
    print('\tTest (Out Of Distribution) acccuracy: %.2f' %(test_acc/len(test_OOD)) + '%')
```

    
    Model blur:
    	Train acccuracy: 65.85%
    	Validation acccuracy: 50.22%
    	Test (Dataloader) acccuracy: 64.64%
    	Test (In Distribution) acccuracy: 35.62%
    	Test (Out Of Distribution) acccuracy: 23.21%
    
    Model ghosting:
    	Train acccuracy: 71.51%
    	Validation acccuracy: 61.56%
    	Test (Dataloader) acccuracy: 71.63%
    	Test (In Distribution) acccuracy: 94.38%
    	Test (Out Of Distribution) acccuracy: 94.20%
    
    Model motion:
    	Train acccuracy: 54.65%
    	Validation acccuracy: 55.18%
    	Test (Dataloader) acccuracy: 51.64%
    	Test (In Distribution) acccuracy: 43.75%
    	Test (Out Of Distribution) acccuracy: 75.00%
    
    Model noise:
    	Train acccuracy: 54.21%
    	Validation acccuracy: 38.33%
    	Test (Dataloader) acccuracy: 45.61%
    	Test (In Distribution) acccuracy: 73.75%
    	Test (Out Of Distribution) acccuracy: 72.32%
    
    Model resolution:
    	Train acccuracy: 73.93%
    	Validation acccuracy: 73.58%
    	Test (Dataloader) acccuracy: 70.89%
    	Test (In Distribution) acccuracy: 86.88%
    	Test (Out Of Distribution) acccuracy: 84.82%
    
    Model spike:
    	Train acccuracy: 65.15%
    	Validation acccuracy: 65.19%
    	Test (Dataloader) acccuracy: 59.21%
    	Test (In Distribution) acccuracy: 88.12%
    	Test (Out Of Distribution) acccuracy: 80.80%