# Homework 2 code
The root folder is composed of 3 elements: `hwk2/   utils/  requirements.txt`.
The data is in `hwk2/data/`.\
In order to run the code, please install the requirements in `requirements.txt` by doing in the root folder:
```$xslt
pip install -r requirements.txt
```
Then run the main script FROM THE ROOT FOLDER:
```$xslt
cd hwk2/
python main.py
```
This will open the different graphs for the different datasets and models. You will need to close one graph in order to see the next one.

## Organisation of the code
The code is written in different classes for different models (EmModel + Kmeans). I also created an `utils/` folder containing all the util methods to process the data (reading file, mathematical tools, data visualization...). This folder will be re-used across different assignments.

## Troubleshooting
There are some known issues regarding the usage of `matplotlib` in a python virtual environment on `OSX` (especially using `virtualenv`). If you get this type of error when running `python main.py`:
```buildoutcfg
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
```
You can read the few workarounds here [Working with Matplotlib on OSX](https://matplotlib.org/faq/osx_framework.html#osxframework-faq). The workaround I use is the following:\
The idea is to use the non virtualenv python along with the `PYTHONHOME` environment variable. Copy the following in your `~/.bashrc` and replace `/usr/bin/python2.7` with your `$PYTHONHOME`:
```buildoutcfg
function frameworkpython {
    if [[ ! -z "$VIRTUAL_ENV" ]]; then
              PYTHONHOME=$VIRTUAL_ENV /usr/bin/python2.7 "$@"
            else
                /usr/bin/python2.7 "$@"
              fi
            }
```
Then to run the main script use `frameworkpython main.py` instead. 
