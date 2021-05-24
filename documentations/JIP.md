# Artefact Classifiers
With this branch, 6 artefact classifiers (blur, ghosting, motion, noise, resolution and spike) can be trained, retrained with new data, tested and used for inference. The classifiers provide information about the quality of the provided CT scans. If the quality is perfect, ie. no artefact is visible in a CT scan, the classifier for the corresponding artefact will return 1. If the artefact is present and the image has a bad quality, the classifier will return 0. Everyhting in between can be interpreted accordingly. For a CT scan with a little bit of blurring in it, the blur classifier would return a quality metric of 0.9 eg.


## Table Of Contents

[Command Line Arguments](#command-line-arguments)

[Preprocessing data](#preprocessing-data)

[Training classifiers](#training-classifiers)

[Retraining classifiers](#retraining-classifiers)

[Testing classifiers](#testing-classifiers)
  * [Test In Distribution](#test-in-distribution)
  * [Test Out Of Distribution](#test-out-of-distribution)
  * [Test In and Out Of Distribution](#test-in-and-out-of-distribution)

[Performing inference](#performing-inference)


## Command Line Arguments
All the provided methods that will be introduced later on, use more or less the same command line arguments. In this section, all arguments are presented, however it is important to note that not every argument is used in each method. The details, ie. which method uses which arguments and how they are used will be discussed in the corresponding section of the method. The following list shows all command line arguments that can be set when executing the `python JIP.py ...` command:


| Tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| `--noise_type` | Specify the CT artefact on which the model will be trained. | no | `blur, ghosting, motion, noise, resolution, spike` | `blur` |
| `--mode` | Specify in which mode to use the model. | yes | `preprocess, train, retrain, testID, testOOD, testIOOD, inference` | -- |
| `--datatype` | Only necessary for `--mode preprocess`. Indicates which data should be preprocessed. | no | `all, train, test, inference` | `all` |
| `--device` | Use this to specify which GPU device to use. | no | `[0, 1, ..., 7]` | `4` |
| `--restore` | Set this for restoring/to continue preprocessing or training. | no | -- | `False` |
| `--store_data` | Store the actual datapoints and save them as .npy after training. | no | -- | `False` |
| `--try_catch_repeat` | Try to train the model with a restored state, if an error occurs. Repeat only <TRY_CATCH_REPEAT> number of times. Can also be used for preprocessing. | no | -- | `False` |
| `--idle_time` | Specify the idle time (waiting time) in seconds after an error occurs before starting the process again when using `--try_catch_repeat`. | if `--try_catch_repeat` is used | -- | `0` |
| `--use_telegram_bot` | Send messages during training through a Telegram Bot (Token and Chat-ID need to be set in mp/paths.py, otherwise an error occurs!). | no | -- | `False` |
| `-h` or `--help` | Simply shows help on which arguments can and should be used. | -- | -- | -- |

For the following sections, it is expected that everything is installed as described in [here](../README.md/#medical_pytorch) and that the commands are executed in the right folder *-- inside medical_pytorch where [JIP.py](../JIP.py) is located --* using the corresponding Anaconda environment, if one is used. The steps before executing any of the introduced commands in the upcoming sections should look *-- more or less --* like the following:
```bash
                  ~ $ cd medical_pytorch
		          ~ $ source ~/.bashrc
		          ~ $ source activate <your_anaconda_env>
<your_anaconda_env> $ python JIP.py ...
```

## Preprocessing data
In order to be able to do inference or training/testing the provided artefact classifiers, the data needs to be preprocessed first. For this, the `--mode` command needs to be set to *preprocess*, whereas the `--datatype` needs to be specified as well. The tags `--device`, `--try_catch_repeat`, `--idle_time` and `--use_telegram_bot` can be set as well *-- if desired --*. The tag `--restore` needs to be used if the preprocessing failed/stopped during the process, so it can be continued where the program stopped, without preprocessing everything from the beginning. So in general the command for preprocessing looks like the following:
```bash
<your_anaconda_env> $ python JIP.py --mode preprocess --datatype <type> --device <GPU_ID> [--restore --try_catch_repeat <nr> --idle_time <time_in_sec> --use_telegram_bot]
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
```bash
<your_anaconda_env> $ python JIP.py --mode train --device <GPU_ID> --datatype train --noise_type <artefact> [--store_data --try_catch_repeat <nr> --idle_time <time_in_sec> --use_telegram_bot --restore]
```

## Retraining classifiers
```bash
<your_anaconda_env> $ python JIP.py --mode retrain --device <GPU_ID> --datatype train --noise_type <artefact> [--store_data --try_catch_repeat <nr> --idle_time <time_in_sec> --use_telegram_bot --restore]
```

## Testing classifiers
--> See provided notebooks

### Test In Distribution
```bash
<your_anaconda_env> $ python JIP.py --mode testID --device <GPU_ID> --datatype test --noise_type <artefact> [--use_telegram_bot --store_data]
```

### Test Out Of Distribution
```bash
<your_anaconda_env> $ python JIP.py --mode testOOD --device <GPU_ID> --datatype test --noise_type <artefact> [--store_data --use_telegram_bot]
```

### Test In and Out Of Distribution
```bash
<your_anaconda_env> $ python JIP.py --mode testIOOD --device <GPU_ID> --datatype test --noise_type <artefact> [--use_telegram_bot --store_data]
```

## Performing inference
For this step, it is important that all 6 artefact classifiers are trained.
```bash
<your_anaconda_env> $ python JIP.py --mode inference --device <GPU_ID> [--use_telegram_bot]
```