## <center> Precision, Recall (Sensitivity, Specificity) and Error calculation for JIP models </center>

This Notebook calculates the Precision, Recall (Sensitivity, Specificity) and Error of the predictions made by the JIP models for the test datasets.
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
import numpy as np
from itertools import zip_longest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report

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

#### Generate a Classification Report including Precision, Recall and F1 Score


```python
print('NOTE: In every confusion matrix: x-axis --> predicted, y-axis --> actual.')
# Loop through the transformed data and calculate everything
for test_name, results in y_yhats.items():
    confusion = confusion_matrix(results['ground_truth'].argmax(axis=1),
                                 results['prediction'].argmax(axis=1),
                                 labels=[0, 1, 2, 3, 4])
    print('\n{}:'.format(test_name))
    print('Confusion Matrix\n')
    print(confusion)
    
    print('\nClassification Report\n')
    print(classification_report(results['prediction'], results['ground_truth'],
                                target_names=['Quality 1', 'Quality 2', 'Quality 3', 'Quality 4', 'Quality 5']))
    
    # Calculate mean values for each element in the loop
    print('\nSummarized Report\n')
    report = classification_report(results['prediction'], results['ground_truth'],
                        target_names=['Quality 1', 'Quality 2', 'Quality 3', 'Quality 4', 'Quality 5'],
                        output_dict=True)
    
    precision, recall, f1 = list(), list(), list()
    idx = 0
    for k, v in report.items():
        if v['support'] != 0 and idx < 5:
            precision.append(v['precision'])
            recall.append(v['recall'])
            f1.append(v['f1-score'])
        idx += 1
    print('Mean precision (without avg values): {}'.format(sum(precision)/len(precision)))
    print('Mean recall (without avg values): {}'.format(sum(recall)/len(recall)))
    print('Mean f1-score (without avg values): {}'.format(sum(f1)/len(f1)))
    print('--------------------------------------------------------------')
    
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    """
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)"""
    
    # Each variable holds the values for every class, so calculate micro avg
    Sensitivity = TP.sum()/ (TP.sum() + FN.sum())
    Specificity = TN.sum()/ (TN.sum() + FP.sum())
    
    print('micro avg -- sensitivity: {}'.format(Sensitivity))
    print('micro avg -- specificity: {}'.format(Specificity))
```

    NOTE: In every confusion matrix: x-axis --> predicted, y-axis --> actual.
    
    test_dl-blur:
    Confusion Matrix
    
    [[195  53   2  33   5]
     [  8 176  21  11   7]
     [ 39  33 157  14  15]
     [  1  16  32  59  19]
     [ 18  22  53  21 190]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.68      0.75      0.71       261
       Quality 2       0.79      0.59      0.67       300
       Quality 3       0.61      0.59      0.60       265
       Quality 4       0.46      0.43      0.45       138
       Quality 5       0.62      0.81      0.70       236
    
       micro avg       0.65      0.65      0.65      1200
       macro avg       0.63      0.63      0.63      1200
    weighted avg       0.66      0.65      0.64      1200
     samples avg       0.65      0.65      0.65      1200
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.6328830124823145
    Mean recall (without avg values): 0.6317733822567451
    Mean f1-score (without avg values): 0.6265583596748644
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.6475
    micro avg -- specificity: 0.911875
    
    test_ID-blur:
    Confusion Matrix
    
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0 23  9 35 13]
     [ 0  6  5 18 19]
     [ 0  0  2  0 30]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         0
       Quality 2       0.00      0.00      0.00        29
       Quality 3       0.11      0.56      0.19        16
       Quality 4       0.38      0.34      0.36        53
       Quality 5       0.94      0.48      0.64        62
    
       micro avg       0.36      0.36      0.36       160
       macro avg       0.29      0.28      0.24       160
    weighted avg       0.50      0.36      0.38       160
     samples avg       0.36      0.36      0.36       160
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.35625
    Mean recall (without avg values): 0.34649840231284235
    Mean f1-score (without avg values): 0.29555837897619547
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.35625
    micro avg -- specificity: 0.8390625
    
    test_OOD-blur:
    Confusion Matrix
    
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0 12  2 40 42]
     [ 0 15  9 92 12]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         0
       Quality 2       0.00      0.00      0.00        27
       Quality 3       0.00      0.00      0.00        11
       Quality 4       0.42      0.30      0.35       132
       Quality 5       0.09      0.22      0.13        54
    
       micro avg       0.23      0.23      0.23       224
       macro avg       0.10      0.11      0.10       224
    weighted avg       0.27      0.23      0.24       224
     samples avg       0.23      0.23      0.23       224
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.12760416666666669
    Mean recall (without avg values): 0.13131313131313133
    Mean f1-score (without avg values): 0.12068633121264702
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.23214285714285715
    micro avg -- specificity: 0.8080357142857143
    
    test_dl-ghosting:
    Confusion Matrix
    
    [[192  62   0   0   3]
     [ 62 175   0   0   3]
     [  0   0 171  21   1]
     [  0   0  71 239   8]
     [ 46  18  24  23  81]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.75      0.64      0.69       300
       Quality 2       0.73      0.69      0.71       255
       Quality 3       0.89      0.64      0.75       266
       Quality 4       0.75      0.84      0.80       283
       Quality 5       0.42      0.84      0.56        96
    
       micro avg       0.71      0.71      0.71      1200
       macro avg       0.71      0.73      0.70      1200
    weighted avg       0.75      0.71      0.72      1200
     samples avg       0.71      0.71      0.71      1200
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.7071412136934498
    Mean recall (without avg values): 0.7314809241717889
    Mean f1-score (without avg values): 0.6998834769702167
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.715
    micro avg -- specificity: 0.92875
    
    test_ID-ghosting:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   8   0   1 151]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         0
       Quality 2       0.00      0.00      0.00         8
       Quality 3       0.00      0.00      0.00         0
       Quality 4       0.00      0.00      0.00         1
       Quality 5       0.94      1.00      0.97       151
    
       micro avg       0.94      0.94      0.94       160
       macro avg       0.19      0.20      0.19       160
    weighted avg       0.89      0.94      0.92       160
     samples avg       0.94      0.94      0.94       160
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.3145833333333333
    Mean recall (without avg values): 0.3333333333333333
    Mean f1-score (without avg values): 0.32368703108252944
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.94375
    micro avg -- specificity: 0.9859375
    
    test_OOD-ghosting:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  1   6   0   6 211]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         1
       Quality 2       0.00      0.00      0.00         6
       Quality 3       0.00      0.00      0.00         0
       Quality 4       0.00      0.00      0.00         6
       Quality 5       0.94      1.00      0.97       211
    
       micro avg       0.94      0.94      0.94       224
       macro avg       0.19      0.20      0.19       224
    weighted avg       0.89      0.94      0.91       224
     samples avg       0.94      0.94      0.94       224
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.23549107142857142
    Mean recall (without avg values): 0.25
    Mean f1-score (without avg values): 0.2425287356321839
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.9419642857142857
    micro avg -- specificity: 0.9854910714285714
    
    test_dl-motion:
    Confusion Matrix
    
    [[137  74   9   6   0]
     [ 86 106  11   4   0]
     [ 44  32  95  58  10]
     [ 20  15  16 138   4]
     [ 33  23  28 109 142]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.61      0.43      0.50       320
       Quality 2       0.51      0.42      0.46       250
       Quality 3       0.40      0.60      0.48       159
       Quality 4       0.72      0.44      0.54       315
       Quality 5       0.42      0.91      0.58       156
    
       micro avg       0.52      0.52      0.52      1200
       macro avg       0.53      0.56      0.51      1200
    weighted avg       0.56      0.52      0.51      1200
     samples avg       0.52      0.52      0.52      1200
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.5309336056902216
    Mean recall (without avg values): 0.5595921850162416
    Mean f1-score (without avg values): 0.5129663791183201
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.515
    micro avg -- specificity: 0.87875
    
    test_ID-motion:
    Confusion Matrix
    
    [[ 0  0  0  0  0]
     [ 0  0  0  0  0]
     [ 0  0  2  2 12]
     [ 0  0  2 14 64]
     [ 0  0  1  9 54]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         0
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.12      0.40      0.19         5
       Quality 4       0.17      0.56      0.27        25
       Quality 5       0.84      0.42      0.56       130
    
       micro avg       0.44      0.44      0.44       160
       macro avg       0.23      0.28      0.20       160
    weighted avg       0.72      0.44      0.50       160
     samples avg       0.44      0.44      0.44       160
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.38125000000000003
    Mean recall (without avg values): 0.4584615384615385
    Mean f1-score (without avg values): 0.3379479626902307
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.4375
    micro avg -- specificity: 0.859375
    
    test_OOD-motion:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   6  50 168]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         0
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.00      0.00      0.00         6
       Quality 4       0.00      0.00      0.00        50
       Quality 5       0.75      1.00      0.86       168
    
       micro avg       0.75      0.75      0.75       224
       macro avg       0.15      0.20      0.17       224
    weighted avg       0.56      0.75      0.64       224
     samples avg       0.75      0.75      0.75       224
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.25
    Mean recall (without avg values): 0.3333333333333333
    Mean f1-score (without avg values): 0.2857142857142857
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.75
    micro avg -- specificity: 0.9375
    
    test_dl-noise:
    Confusion Matrix
    
    [[175  36  20   8   1]
     [122  67  49  14   5]
     [ 57  19  88  45  30]
     [ 46  20  36 101  53]
     [ 30  20  20  22 116]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.73      0.41      0.52       430
       Quality 2       0.26      0.41      0.32       162
       Quality 3       0.37      0.41      0.39       213
       Quality 4       0.39      0.53      0.45       190
       Quality 5       0.56      0.57      0.56       205
    
       micro avg       0.46      0.46      0.46      1200
       macro avg       0.46      0.47      0.45      1200
    weighted avg       0.52      0.46      0.47      1200
     samples avg       0.46      0.46      0.46      1200
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.4620582900568233
    Mean recall (without avg values): 0.46622702738214733
    Mean f1-score (without avg values): 0.44924715989959807
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.4558333333333333
    micro avg -- specificity: 0.8639583333333334
    
    test_ID-noise:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  2   0   1   5  24]
     [  0   0   1  14 113]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         2
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.00      0.00      0.00         2
       Quality 4       0.16      0.26      0.20        19
       Quality 5       0.88      0.82      0.85       137
    
       micro avg       0.74      0.74      0.74       160
       macro avg       0.21      0.22      0.21       160
    weighted avg       0.77      0.74      0.75       160
     samples avg       0.74      0.74      0.74       160
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.259765625
    Mean recall (without avg values): 0.2719938532462543
    Mean f1-score (without avg values): 0.2622271550129486
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.7375
    micro avg -- specificity: 0.934375
    
    test_OOD-noise:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  2   0   1   3  42]
     [  0   0   0  17 159]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         2
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.00      0.00      0.00         1
       Quality 4       0.06      0.15      0.09        20
       Quality 5       0.90      0.79      0.84       201
    
       micro avg       0.72      0.72      0.72       224
       macro avg       0.19      0.19      0.19       224
    weighted avg       0.82      0.72      0.76       224
     samples avg       0.72      0.72      0.72       224
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.24147727272727273
    Mean recall (without avg values): 0.23526119402985074
    Mean f1-score (without avg values): 0.2329341550943985
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.7232142857142857
    micro avg -- specificity: 0.9308035714285714
    
    test_dl-resolution:
    Confusion Matrix
    
    [[181   3   5   5  30]
     [  2 220  80   4   1]
     [  8  12 153  84  13]
     [  0   0   5 166   6]
     [ 44  29   1  17 131]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.81      0.77      0.79       235
       Quality 2       0.72      0.83      0.77       264
       Quality 3       0.57      0.63      0.60       244
       Quality 4       0.94      0.60      0.73       276
       Quality 5       0.59      0.72      0.65       181
    
       micro avg       0.71      0.71      0.71      1200
       macro avg       0.72      0.71      0.71      1200
    weighted avg       0.74      0.71      0.71      1200
     samples avg       0.71      0.71      0.71      1200
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.7238515912474534
    Mean recall (without avg values): 0.7111602922116632
    Mean f1-score (without avg values): 0.7075191196846656
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.7091666666666666
    micro avg -- specificity: 0.9272916666666666
    
    test_ID-resolution:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   1  15]
     [  0   0   0   0   0]
     [  3   0   0   2 139]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         3
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.00      0.00      0.00         0
       Quality 4       0.00      0.00      0.00         3
       Quality 5       0.97      0.90      0.93       154
    
       micro avg       0.87      0.87      0.87       160
       macro avg       0.19      0.18      0.19       160
    weighted avg       0.93      0.87      0.90       160
     samples avg       0.87      0.87      0.87       160
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.32175925925925924
    Mean recall (without avg values): 0.3008658008658009
    Mean f1-score (without avg values): 0.3109619686800895
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.86875
    micro avg -- specificity: 0.9671875
    
    test_OOD-resolution:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  5   0   0  29 190]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         5
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.00      0.00      0.00         0
       Quality 4       0.00      0.00      0.00        29
       Quality 5       0.85      1.00      0.92       190
    
       micro avg       0.85      0.85      0.85       224
       macro avg       0.17      0.20      0.18       224
    weighted avg       0.72      0.85      0.78       224
     samples avg       0.85      0.85      0.85       224
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.28273809523809523
    Mean recall (without avg values): 0.3333333333333333
    Mean f1-score (without avg values): 0.3059581320450886
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.8482142857142857
    micro avg -- specificity: 0.9620535714285714
    
    test_dl-spike:
    Confusion Matrix
    
    [[193  16   8   5   4]
     [  0 181  25  30   4]
     [  0 148  98  51   5]
     [  0  63  12 115  49]
     [ 14  29   6  19 125]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.85      0.93      0.89       207
       Quality 2       0.75      0.41      0.53       437
       Quality 3       0.32      0.66      0.43       149
       Quality 4       0.48      0.52      0.50       220
       Quality 5       0.65      0.67      0.66       187
    
       micro avg       0.59      0.59      0.59      1200
       macro avg       0.61      0.64      0.60      1200
    weighted avg       0.65      0.59      0.60      1200
     samples avg       0.59      0.59      0.59      1200
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.6122984441418878
    Mean recall (without avg values): 0.6390898768345307
    Mean f1-score (without avg values): 0.6039481583623503
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.5933333333333334
    micro avg -- specificity: 0.8983333333333333
    
    test_ID-spike:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [ 13   0   1   5 141]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00        13
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.00      0.00      0.00         1
       Quality 4       0.00      0.00      0.00         5
       Quality 5       0.88      1.00      0.94       141
    
       micro avg       0.88      0.88      0.88       160
       macro avg       0.18      0.20      0.19       160
    weighted avg       0.78      0.88      0.83       160
     samples avg       0.88      0.88      0.88       160
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.2203125
    Mean recall (without avg values): 0.25
    Mean f1-score (without avg values): 0.23421926910299
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.88125
    micro avg -- specificity: 0.9703125
    
    test_OOD-spike:
    Confusion Matrix
    
    [[  0   0   0   0   0]
     [  0   0   0   0   0]
     [  0   0   0   0   0]
     [  3   0   0   2  11]
     [  6   0   1  22 179]]
    
    Classification Report
    
                  precision    recall  f1-score   support
    
       Quality 1       0.00      0.00      0.00         9
       Quality 2       0.00      0.00      0.00         0
       Quality 3       0.00      0.00      0.00         1
       Quality 4       0.12      0.08      0.10        24
       Quality 5       0.86      0.94      0.90       190
    
       micro avg       0.81      0.81      0.81       224
       macro avg       0.20      0.21      0.20       224
    weighted avg       0.74      0.81      0.77       224
     samples avg       0.81      0.81      0.81       224
    
    
    Summarized Report
    
    Mean precision (without avg values): 0.24639423076923078
    Mean recall (without avg values): 0.256359649122807
    Mean f1-score (without avg values): 0.24987437185929648
    --------------------------------------------------------------
    micro avg -- sensitivity: 0.8080357142857143
    micro avg -- specificity: 0.9520089285714286


#### Calculate the error based on absolute difference between y and yhat (Mean Absolute Error)

First of all, the Mean Absolute Error is calculated, which is simply the absolute error between the ground truth and the predicted quality.

Example:

* Ground truth for the quality of three images is $$y = [2, 5, 2]$$
* Model predictions: $$\hat y = [1, 3, 5]$$
* MAE would be: $$MAE(y, \hat y) = \frac{\sum_{i=0}^N |y_{i} - \hat y_{i}|}{N} = \frac{|2 - 1| + |5 - 3| + |2 - 5|}{3} = \frac{6}{3} = 2$$


```python
print("MAE results for each test:\n")
# Loop through the transformed data and calculate MAE
for test_name, results in y_yhats.items():
    print('{}: {:.2f}'.format(test_name, mean_absolute_error(results['prediction'].argmax(axis=1),
                                                             results['ground_truth'].argmax(axis=1))))
```

    MAE results for each test:
    
    test_dl-blur: 0.63
    test_ID-blur: 0.78
    test_OOD-blur: 1.00
    test_dl-ghosting: 0.46
    test_ID-ghosting: 0.16
    test_OOD-ghosting: 0.12
    test_dl-motion: 0.74
    test_ID-motion: 0.64
    test_OOD-motion: 0.28
    test_dl-noise: 0.89
    test_ID-noise: 0.29
    test_OOD-noise: 0.29
    test_dl-resolution: 0.56
    test_ID-resolution: 0.28
    test_OOD-resolution: 0.22
    test_dl-spike: 0.61
    test_ID-spike: 0.37
    test_OOD-spike: 0.30



```python
print("MAE results for each model:\n")
# Loop through the transformed data and calculate MAE
for (t1, results1), (t2, results2), (t3, results3) in grouper(3, y_yhats.items()):
    model = t1.split('-')[1]
    avg_mae = mean_absolute_error(results1['prediction'].argmax(axis=1),
                                  results1['ground_truth'].argmax(axis=1))
    avg_mae += mean_absolute_error(results2['prediction'].argmax(axis=1),
                                   results2['ground_truth'].argmax(axis=1))
    avg_mae += mean_absolute_error(results3['prediction'].argmax(axis=1),
                                   results3['ground_truth'].argmax(axis=1))
    
    print('Model {}: {:.2f}'.format(model, avg_mae/3))
```

    MAE results for each model:
    
    Model blur: 0.80
    Model ghosting: 0.25
    Model motion: 0.55
    Model noise: 0.49
    Model resolution: 0.35
    Model spike: 0.43


In a next step, we want an error metric, that shows how far off the model predicts on average. For this, we use the MAE and divide it by the possible number of (quality) classes - 1, here $5 - 1 = 4$. For instance this means that a higher error of two model indicates the predictions of the one model with higher error are less good than the one with the smaller error. So the higher the error, the far off the predictions are from the ground truth values. Maximum error is 100% and would be the case if all images have a quality of 5 and the model predicts a quality of 1 for every image, assuming that the possible values are $(1, 2, 3, 4, 5)$, ie. 5 in total. However the greatest distance a model can achieve is 4 (predict 1 and actual is 5) --> very bad. Let's have a look at a small example:

* Ground truth for the quality of three images is $$y = [2, 5, 2]$$
* Model one makes the following predictions: $$\hat y^{1} = [1, 3, 5]$$
* Model two makes the following predictions: $$\hat y^{2} = [2, 4, 3]$$

Obviously one might correctly assume that model two makes better predictions than model one. Likewise, the error for model two is lower than model one:

* Error for model one: $$\frac{MAE(y, \hat y^{1})}{4} = \frac{2}{4} \Rightarrow 50\%$$ 
* Error for model two: $$\frac{MAE(y, \hat y^{2})}{4} = \frac{0.667}{4} \Rightarrow 16. 67\%$$


```python
print("Error results for each test:\n")

# Loop through the transformed data and calculate the error by dividing by number_classes
for test_name, results in y_yhats.items():
    print('{}: {:.2f}%'.format(test_name, 100*(mean_absolute_error(results['prediction'].argmax(axis=1),
                                                                   results['ground_truth'].argmax(axis=1))/4)))
```

    Error results for each test:
    
    test_dl-blur: 15.71%
    test_ID-blur: 19.38%
    test_OOD-blur: 24.89%
    test_dl-ghosting: 11.58%
    test_ID-ghosting: 3.91%
    test_OOD-ghosting: 3.12%
    test_dl-motion: 18.52%
    test_ID-motion: 16.09%
    test_OOD-motion: 6.92%
    test_dl-noise: 22.19%
    test_ID-noise: 7.34%
    test_OOD-noise: 7.37%
    test_dl-resolution: 14.00%
    test_ID-resolution: 7.03%
    test_OOD-resolution: 5.47%
    test_dl-spike: 15.21%
    test_ID-spike: 9.22%
    test_OOD-spike: 7.59%



```python
print("Average error results for each model:\n")

# Loop through the transformed data and calculate MAE
for (t1, results1), (t2, results2), (t3, results3) in grouper(3, y_yhats.items()):
    model = t1.split('-')[1]
    error = mean_absolute_error(results1['prediction'].argmax(axis=1),
                                results1['ground_truth'].argmax(axis=1))/4
    error += mean_absolute_error(results2['prediction'].argmax(axis=1),
                                 results2['ground_truth'].argmax(axis=1))/4
    error += mean_absolute_error(results3['prediction'].argmax(axis=1),
                                 results3['ground_truth'].argmax(axis=1))/4
    
    print('Model {}: {:.2f}%'.format(model, 100*error/3))
```

    Average error results for each model:
    
    Model blur: 19.99%
    Model ghosting: 6.20%
    Model motion: 13.84%
    Model noise: 12.30%
    Model resolution: 8.83%
    Model spike: 10.67%



```python
print("Standard deviation of predictions based on actual labels:\n")

# Loop through the transformed data and calculate the std
for test_name, results in y_yhats.items():
    std = np.std(np.abs(results['prediction'].argmax(axis=1) - results['ground_truth'].argmax(axis=1)))
    print('{}: {:.2f}'.format(test_name, std))
```

    Standard deviation of predictions based on actual labels:
    
    test_dl-blur: 1.00
    test_ID-blur: 0.66
    test_OOD-blur: 0.77
    test_dl-ghosting: 0.94
    test_ID-ghosting: 0.66
    test_OOD-ghosting: 0.57
    test_dl-motion: 0.97
    test_ID-motion: 0.63
    test_OOD-motion: 0.50
    test_dl-noise: 1.04
    test_ID-noise: 0.54
    test_OOD-noise: 0.51
    test_dl-resolution: 1.10
    test_ID-resolution: 0.78
    test_OOD-resolution: 0.66
    test_dl-spike: 0.89
    test_ID-spike: 1.10
    test_OOD-spike: 0.79

