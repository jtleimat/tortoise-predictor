# tortoise-predictor

This model and repository is a work in progress! The current model may be changed and this text file will need to be updated with more specific instructions. Current contents you will find include the model, the requirements script, the utility script, and the script to run the tortoise predictor. You will need to download all of these files. The model will likely need to be downloaded separately due to its large size. If the downloaded model says 0 bytes, it did not download properly.

Prior to running this script, you will need to download python to your device. [You can download Python here](https://www.python.org/downloads/).

Next, create a .txt file that lists the location of all images you want to run through the tortoise predictor. To do this, in mac terminal, you will use the following command:

`find [path to folder containing images] -type f | sort > imagelist.txt`

You will need to create a virtual environment. This will allow you to download the packages to run the predictor without changing anything on your personal computer. To do this follow these steps:


Then, run the virtual environment and install the packages from the requirements file.



Within the tortoise predictor, there are several changeable factors. Most important is the display images. The default is set to False because it will bog down your computer to try and display each of these images. Instead, the output will be a txt file that lists all of the images that the model predicts a tortoise to be in and how confident the model is in these predictions.


Thank you and feel free to email if you have questions (jackietleimat@gmail.com)
