# medical_pytorch

To install with Anaconda:
0. (Create a Python3.8 environment, e.g. as conda create -n <env_name> python=3.8, and activate)
2. Install CUDA if not already done and install PyTorch through conda with the command specified by https://pytorch.org/. The tutorial was written using PyTorch 1.6.0. and CUDA10.2., and the command for Linux was at the time 'conda install pytorch torchvision cudatoolkit=10.2 -c pytorch'
3. cd to the project root (where setup.py lives)
4. Execute 'pip install -r requirements.txt'
5. Set paths in mp.paths.py
6. Execute git update-index --assume-unchanged mp/paths.py so that changes in the paths file are not tracked
7. Execute 'pytest' to test the correct installation. Note that one of the tests will test whether at least one GPU is present, if you do not wish to test this mark to ignore. The same holds for tests that used datasets which much be previously downloaded.
8. Please install SimpleCRF, this library is not listed in the requirements as it requires the compilation of C++ code and the process can be tedious:
    - You can first try `pip install SimpleCRF` (this has issue on Windows)
    - If the previous command fails and you have all the proper CUDA dependencies installed:
        - `git clone https://github.com/HiLab-git/SimpleCRF/`
        - Download the latest release of the Eigen C++ library at `https://gitlab.com/libeigen/eigen/-/releases/`
        - Replace the `Eigen` folder found at `SimpleCRF/dependency/densecrf` AND `SimpleCRF/dependency/densedrf3d` with the Eigen folder found in the .zip file you just downloaded
        - Then manually build and install the SimpleCRF library (`cd SimpleCRF`, `python setup.py build`, `python setup.py install`)

When using pylint, unnecessary torch and numpy warnings appear. To avoid, include generated-members=numpy.*,torch.* in the .pylintrc file.


Please take into account the style conventions in style_conventions.py
