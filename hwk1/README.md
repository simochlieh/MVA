# Homework 1 code
In order to run the code, please install the requirements in `requirements.txt` by doing
```
pip install -r requirements.txt
```
Then run the main script FROM THE ROOT FOLDER:
```
cd hwk1/
python main.py
```
This will open the different graphs for the different datasets and models. You will need to close one graph in order to see the next one.

## Organisation of the code
The code is written in different classes for different models. There is 1 parent class BaseClassification` that contains the common methods. Also we created an util folder containing all the util methods to process the data (reading file etc...)

## Troubleshooting
There are some known issues regarding the usage of `matplotlib` in a python virtual environment on `OSX` (especially using `virtualenv`). If you get this type of error when running `python main.py`:
```
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
```
You can read the few workarounds here [Working with Matplotlib on OSX](https://matplotlib.org/faq/osx_framework.html#osxframework-faq). The workaround I use is the following:\
The idea is to use the non virtualenv python along with the `PYTHONHOME` environment variable. Copy the following in your `~/.bashrc` and replace `/usr/bin/python2.7` with your `$PYTHONHOME`:
```
function frameworkpython {
    if [[ ! -z "$VIRTUAL_ENV" ]]; then
              PYTHONHOME=$VIRTUAL_ENV /usr/bin/python2.7 "$@"
            else
                /usr/bin/python2.7 "$@"
              fi
            }
```
Then to run the main script use `frameworkpython main.py` instead. 
