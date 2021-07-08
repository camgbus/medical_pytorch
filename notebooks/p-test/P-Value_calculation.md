## <center> P-value calculation for JIP models </center>

This Notebook calculates the p-values for predictions made by the JIP models for the test datasets.
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
python JIP.py --mode train --device <cuda_id> --datatype train 
                 --noise_type <noise_model> --store_data
```
3. Perform the testing as follows:
```bash
python JIP.py --mode testIOOD --device <cuda_id> --datatype test
                 --noise_type <noise_model> --store_data
```


Once this is finished, everything is set up to run the Notebook.

#### Import necessary libraries


```python
import warnings
warnings.filterwarnings('ignore')

import os
import math
import numpy as np
from sklearn.metrics import confusion_matrix

# -- Grouper from https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python -- #
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
```

#### Set necessary directories
Specify the train_base and test_base directory. These are just the full paths to the JIP folder train_dirs and test_dirs output, for instance: `../JIP/train_dirs/output` and `../JIP/test_dirs/output`.


```python
# Set the base path to JIP/train_dirs/output folder
train_base = '<path>/JIP/train_dirs/output/'
# Set the base path to JIP/test_dirs/output folder
test_base = '<path>/JIP/test_dirs/output/'
```

#### Load data


```python
artefacts = ['blur', 'ghosting', 'motion', 'noise', 'resolution', 'spike']

data = dict()
for artefact in artefacts:
    # Load data
    dl = np.load(os.path.join(train_base, artefact, 'results', 'accuracy_detailed_test.npy'))
    ID = np.load(os.path.join(test_base, artefact, 'testID_results', 'accuracy_detailed_test.npy'))
    OOD = np.load(os.path.join(test_base, artefact, 'testOOD_results', 'accuracy_detailed_test.npy'))
    
    # Create One Hot vectors from predicted values
    for idx, a in enumerate(dl):
        b = np.zeros_like(a[1])
        b[a[1].argmax()] = 1
        a[1] = b
        dl[idx] = a
    for idx, a in enumerate(ID):
        b = np.zeros_like(a[1])
        b[a[1].argmax()] = 1
        a[1] = b
        ID[idx] = a
    for idx, a in enumerate(OOD):
        b = np.zeros_like(a[1])
        b[a[1].argmax()] = 1
        a[1] = b
        OOD[idx] = a
        
    # Save data in dictionary
    data['test_dl-' + artefact] = dl
    data['test_ID-' + artefact] = ID
    data['test_OOD-' + artefact] = OOD
```

#### Transform data into Confusion Matrix


```python
# Transform the data into right format for calculations
y_yhats = dict()
# Loop through all data sets
for k, v in data.items():
    y_yhats[k] = dict()
    y_yhats[k]['prediction'] = list()
    y_yhats[k]['ground_truth'] = list()
    # Change the format of y_yhats --> split in prediction and GT
    for y_yhat in v:
        y_yhats[k]['prediction'].append(y_yhat[1])
        y_yhats[k]['ground_truth'].append(y_yhat[0])
    y_yhats[k]['prediction'] = np.array(y_yhats[k]['prediction'])
    y_yhats[k]['ground_truth'] = np.array(y_yhats[k]['ground_truth'])
```

#### Calculate p-value for all test sets based on the assumption that bad quality images (labels 1, 2 and 3 (and 4)) are really rejected by each classifier


```python
print('NOTE: In every confusion matrix: x-axis --> predicted, y-axis --> actual.')

def p_test(p,y,n):
    '''takes a p value for the null hypothesis, the number of 
    successfull results and the number of unseccessfull results'''
    prob = 0
    combined = y + n 
    for times in np.arange(y,combined+1,1):
        term = p**times * (1-p)**(combined-times)
        bin = math.comb(combined,times)
        prob += term * bin
    return prob

def find_p_p_test(y,n):
    for p in np.arange(1,0,-0.01):
        #print(p,p_test(p,y,n))
        if p_test(p,y,n) <= 0.05:
            return round(p, 2), round(p_test(p,y,n), 2)

# Loop through the transformed data and calculate everything
for test_name, results in y_yhats.items():
    confusion = confusion_matrix(results['ground_truth'].argmax(axis=1),
                                 results['prediction'].argmax(axis=1),
                                 labels=[0, 1, 2, 3, 4])
    print('\n{}:'.format(test_name))
    print('Confusion Matrix')
    print(confusion)
    
    # Compress the confusion metric
    #print()
    #print(confusion[:3, :3]) #TP
    #print(confusion[3:, 3:]) #TN
    #print(confusion[3:, :3]) #FP
    #print(confusion[:3, 3:]) #FN
    compr_confusion = np.array([[confusion[:3, :3].sum(), confusion[:3, 3:].sum()],
                               [confusion[3:, :3].sum(), confusion[3:, 3:].sum()]])
    print('\nCompressed Confusion Matrix')
    print(compr_confusion)
    
    print('\nsensitivity: (probability, p-value) for which we can reject H0: {}'.format(find_p_p_test(confusion[:3, :3].sum(), confusion[:3, 3:].sum())))
    print('specificity: (probability, p-value) for which we can reject H0: {}\n'.format(find_p_p_test(confusion[3:, 3:].sum(), confusion[3:, :3].sum())))
```

    NOTE: In every confusion matrix: x-axis --> predicted, y-axis --> actual.
    
    test_dl-blur:
    Confusion Matrix
    [[195  53   2  33   5]
     [  8 176  21  11   7]
     [ 39  33 157  14  15]
     [  1  16  32  59  19]
     [ 18  22  53  21 190]]
    
    Compressed Confusion Matrix
    [[684  85]
     [142 289]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.86, 0.01)
    specificity: (probability, p-value) for which we can reject H0: (0.63, 0.04)
    
    
    test_ID-blur:
    Confusion Matrix
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0 23  9 35 13]
     [ 0  6  5 18 19]
     [ 0  0  2  0 30]]
    
    Compressed Confusion Matrix
    [[32 48]
     [13 67]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.3, 0.04)
    specificity: (probability, p-value) for which we can reject H0: (0.75, 0.04)
    
    
    test_OOD-blur:
    Confusion Matrix
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0 12  2 40 42]
     [ 0 15  9 92 12]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [ 38 186]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.78, 0.04)
    
    
    test_dl-ghosting:
    Confusion Matrix
    [[192  62   0   0   3]
     [ 62 175   0   0   3]
     [  0   0 171  21   1]
     [  0   0  71 239   8]
     [ 46  18  24  23  81]]
    
    Compressed Confusion Matrix
    [[662  28]
     [159 351]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.94, 0.02)
    specificity: (probability, p-value) for which we can reject H0: (0.65, 0.04)
    
    
    test_ID-ghosting:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   8   0   1 151]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [  8 152]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.91, 0.04)
    
    
    test_OOD-ghosting:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  1   6   0   6 211]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [  7 217]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.94, 0.04)
    
    
    test_dl-motion:
    Confusion Matrix
    [[137  74   9   6   0]
     [ 86 106  11   4   0]
     [ 44  32  95  58  10]
     [ 20  15  16 138   4]
     [ 33  23  28 109 142]]
    
    Compressed Confusion Matrix
    [[594  78]
     [135 393]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.86, 0.04)
    specificity: (probability, p-value) for which we can reject H0: (0.71, 0.04)
    
    
    test_ID-motion:
    Confusion Matrix
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0  0  2  2 12]
     [ 0  0  2 14 64]
     [ 0  0  1  9 54]]
    
    Compressed Confusion Matrix
    [[  2  14]
     [  3 141]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.02, 0.04)
    specificity: (probability, p-value) for which we can reject H0: (0.94, 0.02)
    
    
    test_OOD-motion:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   6  50 168]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [  6 218]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.94, 0.02)
    
    
    test_dl-noise:
    Confusion Matrix
    [[175  36  20   8   1]
     [122  67  49  14   5]
     [ 57  19  88  45  30]
     [ 46  20  36 101  53]
     [ 30  20  20  22 116]]
    
    Compressed Confusion Matrix
    [[633 103]
     [172 292]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.83, 0.02)
    specificity: (probability, p-value) for which we can reject H0: (0.59, 0.05)
    
    
    test_ID-noise:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  2   0   1   5  24]
     [  0   0   1  14 113]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [  4 156]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.94, 0.03)
    
    
    test_OOD-noise:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  2   0   1   3  42]
     [  0   0   0  17 159]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [  3 221]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.96, 0.02)
    
    
    test_dl-resolution:
    Confusion Matrix
    [[181   3   5   5  30]
     [  2 220  80   4   1]
     [  8  12 153  84  13]
     [  0   0   5 166   6]
     [ 44  29   1  17 131]]
    
    Compressed Confusion Matrix
    [[664 137]
     [ 79 320]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.8, 0.02)
    specificity: (probability, p-value) for which we can reject H0: (0.76, 0.03)
    
    
    test_ID-resolution:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   1  15]
     [  0   0   0   0   0]
     [  3   0   0   2 139]]
    
    Compressed Confusion Matrix
    [[  0  16]
     [  3 141]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.94, 0.02)
    
    
    test_OOD-resolution:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  5   0   0  29 190]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [  5 219]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.95, 0.03)
    
    
    test_dl-spike:
    Confusion Matrix
    [[193  16   8   5   4]
     [  0 181  25  30   4]
     [  0 148  98  51   5]
     [  0  63  12 115  49]
     [ 14  29   6  19 125]]
    
    Compressed Confusion Matrix
    [[669  99]
     [124 308]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.84, 0.01)
    specificity: (probability, p-value) for which we can reject H0: (0.67, 0.03)
    
    
    test_ID-spike:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [ 13   0   1   5 141]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [ 14 146]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.86, 0.03)
    
    
    test_OOD-spike:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  3   0   0   2  11]
     [  6   0   1  22 179]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [ 10 214]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.92, 0.03)
    



```python
# Loop through the transformed data and calculate everything
for test_name, results in y_yhats.items():
    confusion = confusion_matrix(results['ground_truth'].argmax(axis=1),
                                 results['prediction'].argmax(axis=1),
                                 labels=[0, 1, 2, 3, 4])
    print('\n{}:'.format(test_name))
    print('Confusion Matrix')
    print(confusion)
    
    # Compress the confusion metric
    #print()
    #print(confusion[:4, :4]) #TP
    #print(confusion[4:, 4:]) #TN
    #print(confusion[4:, :4]) #FP
    #print(confusion[:4, 4:]) #FN
    compr_confusion = np.array([[confusion[:4, :4].sum(), confusion[:4, 4:].sum()],
                               [confusion[4:, :4].sum(), confusion[4:, 4:].sum()]])
    print('\nCompressed Confusion Matrix')
    print(compr_confusion)
    
    print('\nsensitivity: (probability, p-value) for which we can reject H0: {}'.format(find_p_p_test(confusion[:4, :4].sum(), confusion[:4, 4:].sum())))
    print('specificity: (probability, p-value) for which we can reject H0: {}\n'.format(find_p_p_test(confusion[4:, 4:].sum(), confusion[4:, :4].sum())))
```

    
    test_dl-blur:
    Confusion Matrix
    [[195  53   2  33   5]
     [  8 176  21  11   7]
     [ 39  33 157  14  15]
     [  1  16  32  59  19]
     [ 18  22  53  21 190]]
    
    Compressed Confusion Matrix
    [[850  46]
     [114 190]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.93, 0.01)
    specificity: (probability, p-value) for which we can reject H0: (0.57, 0.03)
    
    
    test_ID-blur:
    Confusion Matrix
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0 23  9 35 13]
     [ 0  6  5 18 19]
     [ 0  0  2  0 30]]
    
    Compressed Confusion Matrix
    [[96 32]
     [ 2 30]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.67, 0.03)
    specificity: (probability, p-value) for which we can reject H0: (0.81, 0.04)
    
    
    test_OOD-blur:
    Confusion Matrix
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0 12  2 40 42]
     [ 0 15  9 92 12]]
    
    Compressed Confusion Matrix
    [[ 54  42]
     [116  12]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.47, 0.04)
    specificity: (probability, p-value) for which we can reject H0: (0.05, 0.03)
    
    
    test_dl-ghosting:
    Confusion Matrix
    [[192  62   0   0   3]
     [ 62 175   0   0   3]
     [  0   0 171  21   1]
     [  0   0  71 239   8]
     [ 46  18  24  23  81]]
    
    Compressed Confusion Matrix
    [[993  15]
     [111  81]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.97, 0.0)
    specificity: (probability, p-value) for which we can reject H0: (0.36, 0.04)
    
    
    test_ID-ghosting:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   8   0   1 151]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [  9 151]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.9, 0.04)
    
    
    test_OOD-ghosting:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  1   6   0   6 211]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [ 13 211]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.9, 0.02)
    
    
    test_dl-motion:
    Confusion Matrix
    [[137  74   9   6   0]
     [ 86 106  11   4   0]
     [ 44  32  95  58  10]
     [ 20  15  16 138   4]
     [ 33  23  28 109 142]]
    
    Compressed Confusion Matrix
    [[851  14]
     [193 142]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.97, 0.01)
    specificity: (probability, p-value) for which we can reject H0: (0.37, 0.02)
    
    
    test_ID-motion:
    Confusion Matrix
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0  0  2  2 12]
     [ 0  0  2 14 64]
     [ 0  0  1  9 54]]
    
    Compressed Confusion Matrix
    [[20 76]
     [10 54]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.14, 0.04)
    specificity: (probability, p-value) for which we can reject H0: (0.74, 0.04)
    
    
    test_OOD-motion:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   6  50 168]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [ 56 168]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.69, 0.03)
    
    
    test_dl-noise:
    Confusion Matrix
    [[175  36  20   8   1]
     [122  67  49  14   5]
     [ 57  19  88  45  30]
     [ 46  20  36 101  53]
     [ 30  20  20  22 116]]
    
    Compressed Confusion Matrix
    [[903  89]
     [ 92 116]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.89, 0.02)
    specificity: (probability, p-value) for which we can reject H0: (0.49, 0.03)
    
    
    test_ID-noise:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  2   0   1   5  24]
     [  0   0   1  14 113]]
    
    Compressed Confusion Matrix
    [[  8  24]
     [ 15 113]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.13, 0.05)
    specificity: (probability, p-value) for which we can reject H0: (0.82, 0.04)
    
    
    test_OOD-noise:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  2   0   1   3  42]
     [  0   0   0  17 159]]
    
    Compressed Confusion Matrix
    [[  6  42]
     [ 17 159]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.05, 0.03)
    specificity: (probability, p-value) for which we can reject H0: (0.85, 0.03)
    
    
    test_dl-resolution:
    Confusion Matrix
    [[181   3   5   5  30]
     [  2 220  80   4   1]
     [  8  12 153  84  13]
     [  0   0   5 166   6]
     [ 44  29   1  17 131]]
    
    Compressed Confusion Matrix
    [[928  50]
     [ 91 131]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.93, 0.01)
    specificity: (probability, p-value) for which we can reject H0: (0.53, 0.04)
    
    
    test_ID-resolution:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   1  15]
     [  0   0   0   0   0]
     [  3   0   0   2 139]]
    
    Compressed Confusion Matrix
    [[  1  15]
     [  5 139]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.92, 0.02)
    
    
    test_OOD-resolution:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  5   0   0  29 190]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [ 34 190]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.8, 0.04)
    
    
    test_dl-spike:
    Confusion Matrix
    [[193  16   8   5   4]
     [  0 181  25  30   4]
     [  0 148  98  51   5]
     [  0  63  12 115  49]
     [ 14  29   6  19 125]]
    
    Compressed Confusion Matrix
    [[945  62]
     [ 68 125]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.92, 0.02)
    specificity: (probability, p-value) for which we can reject H0: (0.58, 0.03)
    
    
    test_ID-spike:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [ 13   0   1   5 141]]
    
    Compressed Confusion Matrix
    [[  0   0]
     [ 19 141]]
    
    sensitivity: (probability, p-value) for which we can reject H0: None
    specificity: (probability, p-value) for which we can reject H0: (0.83, 0.05)
    
    
    test_OOD-spike:
    Confusion Matrix
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  3   0   0   2  11]
     [  6   0   1  22 179]]
    
    Compressed Confusion Matrix
    [[  5  11]
     [ 29 179]]
    
    sensitivity: (probability, p-value) for which we can reject H0: (0.13, 0.05)
    specificity: (probability, p-value) for which we can reject H0: (0.81, 0.03)
    

