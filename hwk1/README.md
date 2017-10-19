# Homework 1 code
In order to run the code, please install the requirements in `requirements.txt` by doing
```$xslt
pip install -r requirements.txt
```
Then run the main script FROM THE ROOT FOLDER:
```$xslt
cd hwk1/
python main.py
```
This will open the different graphs for the different datasets and models. You will need to close one graph in order to see the next one.

## Organisation of the code
The code is written in different classes for different models. There is 1 parent class BaseClassification` that contains the common methods. Also we created an util folder containing all the util methods to process the data (reading file etc...)