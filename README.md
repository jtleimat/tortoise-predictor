# tortoise-predictor

This model and repository is a work in progress! The current model may be changed and this text file will need to be updated with more specific instructions. Current contents you will find include the model, the requirements script, the utility script, and the script to run the tortoise predictor. You will need to download all of these files. The model will likely need to be downloaded separately due to its large size. If the downloaded model says 0 bytes, it did not download properly.

Step 1. Prior to running this script, you will need to download python to your device. [You can download Python here](https://www.python.org/downloads/).

Then, you'll need to download all of the files in this repository. I'd recommend moving all of these files onto the desktop of your computer. The instructions will assume you have moved a folder of all files onto your desktop in a folder named 'Tortoise'

Step 2. Open terminal (on mac) or command line (on windows) and set your working directory. This will be where you moved the files from this repository.

On Mac:
`cd Desktop/Tortoise`

On Windows:
`cd Desktop\Tortoise`

Step 3. Create a .txt file that lists the location of all images you want to run through the tortoise predictor. To do this, in mac terminal, you will use the following command:

`find [path to folder containing images] -type f | sort > imagelist.txt`

NOTE: If the images are not in the 'Tortoise' folder, you will need to add the folder path before the file name. Example: IMG_001.png, if in the 'Tortoise' folder, is fine as is.

But, if it is in a folder in Documents (/Documents/CameraTraps/Cam1) you need to add this before each image file. In the text file, each image would need to look like /Documents/CameraTraps/Cam1/IMG_001.png 

Step 4. Create a virtual environment. This will allow you to download the packages to run the predictor without changing anything on your personal computer. And only needs to be done once. If you are running through another set of images, you can skip this step after the first time. To do this follow these steps:

On Mac:
`python3 -m venv venv_name`

On Windows:
`py -m venv venv_name`

Step 5. Run the virtual environment. This needs to be done every time you run the tortoise predictor.

On Mac:
`source venv_name/bin/activate`

On Windows:
`venv_name\Scripts\activate`

Step 6. Install the required package. This only needs to be done once. 

On Mac and Windows:
`pip3 install -r requirements.txt`

Step 7. Run the tortoise predictor.

On Mac:
`python3 tortoise_predictor.py`

On Windows:
`py tortoise_predictor.py`

The predicted tortoises will be put in a text file titled modelresults.txt
Note, if you are running this more than once, you will need to change the name of the output file. To do this add -o after the tortoise predictor and follow the -o with a new txt file name. Example:

On Mac:
`python3 tortoise_predictor.py -o modelresults2.txt`

On Windows:
`py tortoise_predictor.py -o modelresults2.txt`

The results are now in modelresults2.txt

Within the tortoise predictor, there are several changeable factors. Most important is the display images. The default is set to False because it will bog down your computer to try and display each of these images. Instead, the output will be a txt file that lists all of the images that the model predicts a tortoise to be in and how confident the model is in these predictions.

If you do want to preview the images, you would swich the setting:

On Mac:
`python3 tortoise_predictor.py -i False`

On Windows:
`py tortoise_predictor.py -i False`

The confidence of the model can also be toggled. It is set to output predictions whenever the model is at least 50% confident there is a tortoise. This number can be raised or lowered. 

Example to switch the confidence to 80%, this is what you would do:

On Mac:
`python3 tortoise_predictor.py -t 0.8`

On Windows:
`py tortoise_predictor.py -t 0.8`

Thank you and feel free to email if you have questions (jackietleimat@gmail.com)
