## BANK PROJECT
#### A machine learning project for AIML Exam of poliba.it

---

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
##### if you have windows os you shouldn't have problems

---
## START PHASE
To start algorithm just run :

```
python3 run_classifiers.py
```
or just run that file from your ide.

---

## CONTENTS
- `output` directory :
In the output directory you will see all images of plots runned in scripts in `.png` format, and a `report.txt` file which include all log seen in console.
- `prepraringdata` file in which we manipuale data (clean, create new features, deleting duplicates ecc..)
- `model` file in which we defined all the classifers for our alghoritms used for training phase
- `run_classifiers` file in which we defined all classifiers runned in test phase, with a report of an evaluation phase
- `dataset` directory in which there is the dataset imported with pandas in `preparingdata.py`
- `util_service` file in which we define common function used in our implementation.
###### Authors:
###### [Federico D'Errico](https://github.com/federicoder) [Mauro Giordano](https://github.com/mgiordano95) [Francesco Palumbo]()
