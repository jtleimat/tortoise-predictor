# tortoise-predictor

Current contents you will find include the model, the requirements script, the utility script, and the script to run the tortoise predictor. You will need to download all of these files. The model will likely need to be downloaded separately due to its large size. If the downloaded model says 0 bytes, it did not download properly. I'd recommend moving all of these files onto the desktop of your computer in one folder. The instructions will assume you have moved a folder of all files onto your desktop in a folder named 'Tortoise'

Step 1. Prior to running this script, you will need to download python to your device. [You can download Python here](https://www.python.org/downloads/).


Step 2. Open terminal (on mac) or command line (on windows) and set your working directory. This will be where you moved the files from this repository.

On Mac:
`cd Desktop/Tortoise`

On Windows:
`cd Desktop\Tortoise`


Step 3. Create a .txt file that lists the location of all images you want to run through the tortoise predictor.

On Mac:
`realpath *.png > imagelist.txt`

On Windows:
`dir /s /b /a-d "*.png" > imagelist.txt`

NOTE: If your images are not .png, you will just change *.png to *.jpg or *.jpeg

If your images are not in the same folder as your working directory, you will need to change the step to:
`realpath [enter relative filepath here] *.png > imagelist.txt`

Example, if my images are in /Users/jtleimat/Documents/CameraTraps/Cam1, I would code as:
`realpath /Users/jtleimat/Documents/CameraTraps/Cam1/ *.png > imagelist.txt`

Step 4. Check that the text file you created is just a list of file paths. Most commonly on terminal, one extra space will be added at the very end of the file. You can just hit delete to remove that extra space.

If your images are not in the working directory, the text file will have been deposited in that folder, so you will need to move it back to the working directory.

Step 5. Create a virtual environment. This will allow you to download the packages to run the predictor without changing anything on your personal computer. And only needs to be done once. If you are running through another set of images, you can skip this step after the first time. To do this follow these steps:

On Mac:
`python3 -m venv venv_name`

On Windows:
`py -m venv venv_name`


Step 6. Run the virtual environment. This needs to be done every time you run the tortoise predictor.

On Mac:
`source venv_name/bin/activate`

On Windows:
`venv_name\Scripts\activate`


Step 7. Install the required package. This only needs to be done once. 

On Mac and Windows:
`pip3 install -r requirements.txt`

Step 8. Run the tortoise predictor.

On Mac:
`python3 tortoise_predictor.py -i True`

On Windows:
`py tortoise_predictor.py -i True`

For each image, you will see "Image: 1" and a 4 below it while it is processing on the screen. When done, you will see 'Predictions Completed' printed on terminal/command line.

The predicted tortoises will be put in a text file default titled modelresults.txt.
The results will look something like:
/Users/jtleimat/Documents/Cameras/testimages/vlcsnap-2024-01-08-12h08m59s877.png
Prediction 1: bbox = tensor([  5.2013, 643.9755, 172.3244, 739.5422]), score = 0.9967920184135437

The first line displays the image in reference.
The second line displays information related to the prediction. If there are multiple predictions in an image, it will list more predictions. If there are no predictions, the file name will not be printed.
bbox is the approximate coordinates of the predicted tortoise on the image, and score is how confident the model is in the prediction. The closer to 1, the more confident it is. To see where the prediction is, I would recommend running the files from this list through the predictor and have the code display the images (more on this later).

Note, if you are running this more than once, you will want to change the name of the output file. To do this add -o after the tortoise predictor and follow the -o with a new txt file name. Example:

On Mac:
`python3 tortoise_predictor.py -o modelresults2.txt`

On Windows:
`py tortoise_predictor.py -o modelresults2.txt`

The results are now in modelresults2.txt

Step 9. Type and enter 'Exit' to leave the virtual environment. And type and enter 'exit' again to finish running code in terminal/command line.

How to view predictions: Within the tortoise predictor, there are several changeable factors. Most important is the display images. The default code skips the images so it will not bog down your computer memory. Instead, the output will be a txt file that lists all of the images that the model predicts a tortoise to be in and how confident the model is in these predictions.

If you want to preview the images, I would recommend doing no more than 50 images at a time. All you need to do is remove the '-i True' to enable the code to display the predictions:

On Mac:
`python3 tortoise_predictor.py`

On Windows:
`py tortoise_predictor.py`

If you do this, a new window will appear opening matplot. You will have to click the 'x' on the new window to progress to each image. The magnifying glass on the bottom will allow you to zoom in on the predictons.


Changing Confidence Levels: The confidence of the model can also be toggled. It is set to output predictions whenever the model is at least 50% confident there is a tortoise. This number can be raised or lowered. 

Example to switch the confidence to 80%, you would add -t 0.8 (or change the 0.8 to whatever confidence threshold you want the model to operate at):

On Mac:
`python3 tortoise_predictor.py -i True -t 0.8`

On Windows:
`py tortoise_predictor.py -i True -t 0.8`



Thank you and feel free to email if you have questions (jackietleimat@gmail.com)
