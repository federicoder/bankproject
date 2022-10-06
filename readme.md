## BANK PROJECT
#### A machine learning project for AIML Exam of poliba.it

First Download repository by git clone via ssh or https.

## INSTALL PHASE:

go to requirements.txt and run the command:

```
pip install -r requirements.txt
```

##### then if you have a linux os run:
```
sudo apt-get install python-tk

```
And then import Tkinter in `preparingdata.py` and `run_classifiers.py`


##### if you have an arch-linux os:
```
sudo pacman -S tk
```
##### if you have windows os you shouldn't have problems, otherwise you have to delete the code:
```
matplotlib.use('TkAgg')
```
##### from `preparingdata.py` and `run_classifiers.py`
## START PHASE
To start algorithm just run :

```
python3 run_classifiers.py
```
or just run that file from your ide.

In the output directory you will see the images of plots runned in scripts.
There are also .dot image that could be convered in a commod software online to convert .dot into .png

###### Authors:
###### Federico D'Errico Mauro Giordano Francesco Palumbo
