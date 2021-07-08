# Artifact Classifiers
With this branch, 6 artifact classifiers (blur, ghosting, motion, noise, resolution and spike) can be trained, retrained with new data, tested and used for inference. The classifiers provide information about the quality of the provided CT scans. If the quality is perfect, ie. no artifact is visible in a CT scan, the classifier for the corresponding artifact will return 1. If the artifact is present and the image has a bad quality, the classifier will return 0. Everyhting in between can be interpreted accordingly. For a CT scan with a little bit of blurring in it, the blur classifier would return a quality metric of 0.9 eg.


## Table Of Contents
[JIP Datastructure](#jip-datastructure)

[Command Line Arguments](#command-line-arguments)

[Preprocessing data](#preprocessing-data)

[Training classifiers](#training-classifiers)

[Retraining classifiers](#retraining-classifiers)

[Testing classifiers](#testing-classifiers)
  * [Test In Distribution](#test-in-distribution)
  * [Test Out Of Distribution](#test-out-of-distribution)
  * [Test In and Out Of Distribution](#test-in-and-out-of-distribution)

[Performing inference](#performing-inference)

## JIP Datastructure
The whole preprocessing, training, retraining, testing and inference is based on the data stored in the following structure:

    ../JIP/
    ├── data_dirs/
    │   ├── input/
    |   │   ├── patient_00
    |   │   |    ├── img
    |   │   |       ├── img.nii.gz
    |   |   ├── ...
    │   ├── ...
    ├── preprocessed_dirs/
    │   ├── ...
    ├── train_dirs/
    │   ├── input/
    |   │   ├── patient_00
    |   │   |    ├── img
    |   │   |       ├── img.nii.gz
    |   │   |    ├── seg
    |   │   |       ├── 001.nii.gz
    |   |   ├── ...
    |   ├── ...
    └── test_dirs/
        ├── input/
        │   ├── patient_00
        │   |    ├── img
        │   |       ├── img.nii.gz
        │   |    ├── seg
        │   |       ├── 001.nii.gz
        |   ├── ...
        ├── ...

The corresponding paths need to be set in [paths.py](../mp/paths.py) before starting any process. For instance, only the `storage_path` variable needs to be set -- in the example above it would be `../JIP`.

The data for inference *-- data_dirs/input --*, training *-- train_dirs/input --* and testing *-- test_dirs/input --* needs to be provided by the user with respect to the previously introduced structure before starting any process. If this is not done properly, neither one of the later presented methods will work properly, since the data will not be found, thus resulting in an error during runtime. The preprocessed folder will be automatically generated during preprocessing and should not be changed by the user. Note that the folders (`patient_00`, etc.) can be named differently, however the name of the corresponding scan needs to be `img.nii.gz`, a Nifti file located in `img/` folder. The corresponding segmentation needs to be named `001.nii.gz`, also a Nifti file but located in the `seq/` folder. If there should exist more than one segmentation for a scan, the segmentations should then be continously named, like `001.nii.gz`, `002.nii.gz` and `003.nii.gz` for three scans.

## Command Line Arguments
All the provided methods that will be introduced later on, use more or less the same command line arguments. In this section, all arguments are presented, however it is important to note that not every argument is used in each method. The details, ie. which method uses which arguments will be shown in the corresponding section of the method. The following list shows all command line arguments that can be set when executing the `python JIP.py ...` command:


| Tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| `--noise_type` | Specify the CT artifact on which the model will be trained. | no | `blur, ghosting, motion, noise, resolution, spike` | `blur` |
| `--mode` | Specify in which mode to use the model. | yes | `preprocess, train, retrain, testID, testOOD, testIOOD, inference` | -- |
| `--datatype` | Only necessary for `--mode preprocess`. Indicates which data should be preprocessed. | no | `all, train, test, inference` | `all` |
| `--device` | Use this to specify which GPU device to use. | no | `[0, 1, ..., 7]` | `4` |
| `--restore` | Set this for restoring/to continue preprocessing or training. | no | -- | `False` |
| `--store_data` | Store the actual datapoints and save them as .npy after training. | no | -- | `False` |
| `--try_catch_repeat` | Try to train the model with a restored state, if an error occurs. Repeat only <TRY_CATCH_REPEAT> number of times. Can also be used for preprocessing. | no | -- | `False` |
| `--idle_time` | Specify the idle time (waiting time) in seconds after an error occurs before starting the process again when using `--try_catch_repeat`. | if `--try_catch_repeat` is used | -- | `0` |
| `--use_telegram_bot` | Send messages during training through a Telegram Bot (Token and Chat-ID need to be set in [`paths.py`](../mp/paths.py), otherwise an error occurs!). | no | -- | `False` |
| `-h` or `--help` | Simply shows help on which arguments can and should be used. | -- | -- | -- |

For the following sections, it is expected that everything is installed as described in [here](../README.md/#medical_pytorch) and that the commands are executed in the right folder *-- inside medical_pytorch where [JIP.py](../JIP.py) is located --* using the corresponding Anaconda environment, if one is used. The steps before executing any of the introduced commands in the upcoming sections should look *-- more or less --* like the following:
```bash
                  ~ $ cd medical_pytorch
		  ~ $ source ~/.bashrc
		  ~ $ source activate <your_anaconda_env>
<your_anaconda_env> $ python JIP.py ...
```

## Preprocessing data
In order to be able to do inference or training/testing the provided artifact classifiers, the data needs to be preprocessed first. For this, the `--mode` command needs to be set to *preprocess*, whereas the `--datatype` needs to be specified as well. The tags `--device`, `--try_catch_repeat`, `--idle_time` and `--use_telegram_bot` can be set as well *-- if desired --*. The tag `--restore` needs to be used if the preprocessing failed/stopped during the process, so it can be continued where the program stopped, without preprocessing everything from the beginning. So in general the command for preprocessing looks like the following:
```bash
<your_anaconda_env> $ python JIP.py --mode preprocess --datatype <type> --device <GPU_ID>
				   [--restore --try_catch_repeat <nr> --idle_time <time_in_sec>
				    --use_telegram_bot]
```
Let's look at some use cases:
1. Preprocess everything *-- train, test and inference data --* from scratch on GPU device 0:
    ```bash
    <your_anaconda_env> $ python JIP.py --mode preprocess --datatype all --device 0
    ```
2. Continue preprocessing for train data on GPU device 3: 
    ```bash
    <your_anaconda_env> $ python JIP.py --mode preprocess --datatype train --device 3 --restore
    ```
3. Preprocess inference data by using GPU device 7 and the Telegram Bot:
    ```bash
    <your_anaconda_env> $ python JIP.py --mode preprocess --datatype inference --device 7 --use_telegram_bot
    ```
4. Preprocess test data by repeating the preprocessing in any case of failing for max. 3 times with a waiting time of 180 seconds in between each attempt. In this case we want to use the default GPU device:
    ```bash
    <your_anaconda_env> $ python JIP.py --mode preprocess --datatype test --try_catch_repeat 3 --idle_time 180
    ```

## Training classifiers
When training classifiers from the beginning, one can simply execute the following command with respect to the possible arguments: 
```bash
<your_anaconda_env> $ python JIP.py --mode train --device <GPU_ID>
				    --datatype train --noise_type <artifact>
				    [--store_data --try_catch_repeat <nr>
				    --idle_time <time_in_sec> --use_telegram_bot --restore]
```
Again, the restore flag can be used in case of an errorenous termination of the code before the actual termination of the training to continue with the training from the last checkpoint.
In the [JIP.py](../JIP.py) file is a config dictionary that can be modified by the programmer as well, in which one can specify if the dataset needs to be augmented or not. Note, that the augmentation is only implemented for the [Grand Challenge](https://covid-segmentation.grand-challenge.org) and [Decathlon Lung Dataset](http://medicaldecathlon.com), since the labels for the transformed files need to be generated based on the manually created labels. For now, if the programmer wants to perform augmentation on a new dataset, this needs to be changed/added in the code, for instance, the [dataset_JIP_cnn.py](../mp/data/datasets/dataset_JIP_cnn.py) needs to be updated. If no augmentation is necessary, only the provided data will be used, ie. the code does not have to be modified at all for this.

## Retraining classifiers
The retraining of classifiers is used for retraining already trained classifiers with new data. This is mainly used by institutes that are provided with pre-trained classifiers and then can retrain them with their own in-house dataset. Those pre-trained classifiers are trained on a mixed dataset including one in-house, the [Grand Challenge](https://covid-segmentation.grand-challenge.org) and [Decathlon Lung Dataset](http://medicaldecathlon.com) dataset whereas data augmentation has been performed on those datasets as well to ensure equally distributed data among all intensity classes during training. To run such a retrain, obviously the corresponding classifier needs to exist and the own in-house data needs to be stored under `JIP/train_dirs/input` and preprocessed as well. The training can then be started using the following command:
```bash
<your_anaconda_env> $ python JIP.py --mode retrain --device <GPU_ID>
				    --datatype train --noise_type <artifact>
				    [--store_data --try_catch_repeat <nr>
				    --idle_time <time_in_sec> --use_telegram_bot --restore]
```

## Testing classifiers
Testing the performance and accuracy of all classifiers can be performed in two ways. An In-Distribution (ID) test as well as an Out-Of_Distribution (OOD) test can be performed. Both tests are based on the dataset names that are used during training, so if new data is used, ie. different datasets, the ID and OOD test needs to be adapted since the selection of the data for the test is based on the dataset name. Since this varies from use case to use case, this needs to be adapted by the programmer in [dataset_JIP_cnn.py](../mp/data/datasets/dataset_JIP_cnn.py) as well when this is desired to use. Notebooks are provided in the [notebooks folder](../notebooks) and can be reloaded after a test is performed. Every Notebook automatically loads the data and performs the corresponding calculations once the paths is set in those Notebooks. So the user only needs to change the path in the Notebooks. In those Notebooks, the accuracy will be calculated, but also confusion matrices will be extracted from the stored data as well.

### Test In Distribution
For performing an ID-test, the following command can be used:
```bash
<your_anaconda_env> $ python JIP.py --mode testID --device <GPU_ID>
				    --datatype test --noise_type <artifact>
				    [--use_telegram_bot --store_data]
```

### Test Out Of Distribution
For performing an OOD-test, the following command can be used:
```bash
<your_anaconda_env> $ python JIP.py --mode testOOD --device <GPU_ID>
				    --datatype test --noise_type <artifact>
				    [--store_data --use_telegram_bot]
```

### Test In and Out Of Distribution
For performing an ID-test followed by an OOD-test without executing both commands, the following command can be used:
```bash
<your_anaconda_env> $ python JIP.py --mode testIOOD --device <GPU_ID>
				    --datatype test --noise_type <artifact>
				    [--use_telegram_bot --store_data]
```

Note: If the Notebooks are used, remember to set the `--store_data` flag when performing any test, since this stored data is crucial for the calculations performed in the Notebooks.

## Performing inference
Performing inference on data is very straight-forward. It is important that all 6 artifact classifiers are trained and do exist in `JIP/data_dirs/persistent`. The provided preprocessed data from `JIP/data_dirs/input` will then be used and the results for every scan per classifier will be stored in a metrics file, `metrics.json`. After the sucessfull termination the file will be located at `JIP/data_dirs/output`. To start the inference, the following command needs to be executed:
```bash
<your_anaconda_env> $ python JIP.py --mode inference --device <GPU_ID>
				   [--use_telegram_bot]
```

After sucessfull termination, the `metrics.json` file is generated and has the following structure:
```
{	
    "patient_00":	{
                        "LFC": true,
                        "blur": 0.5,
                        "ghosting": 0.72,
                        "motion": 0.98,
                        "noise": 0.67,
                        "resolution": 0.82,
                        "spike": 0.96
                    },
    ...
}
```
For every patient folder in `JIP/data_dirs/input` the same 6 quality metrics will be calculated as well as the `Lung Fully Captured (LFC)` metric, which indicates if the lung in a CT scan is fully captured or if parts of the lungs are missing.
