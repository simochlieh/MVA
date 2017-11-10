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