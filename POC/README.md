**Abstract Summarization** 

**Prerequisite packages** -

    1) Python 3.6.6 - https://www.python.org/downloads/release/python-366/
    2) PyCharm 2018.2 - https://download.jetbrains.com/python/pycharm-community-2018.2.3.exe
    
**Install dependencies**

    python setup.py install
        
**Steps to download data**

    1) Navigate to https://cs.nyu.edu/~kcho/DMQA/ - download 'stories' from CNN dataset
    2) Unzip the file.
    3) Update the paths in config.py as per you machine.
    
**Steps to train model**

    1) Run data_preprocessing.py (make sure paths are corrected in config.py)
    2) Run train_model.py
