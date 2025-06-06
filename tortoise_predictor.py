# Tortoise Predictor

import os
import torch
import utils
from utils import (
    get_model_instance_segmentation,
)

from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib import colormaps
from pathlib import Path
from PIL import Image
import torchvision

parser = OptionParser()
parser.add_option("-i", "--skip_image",
        help="Skip showing each result image",
        action="store_true", default=True)
parser.add_option("-p", "--path_to_images",
        help="Path to text file of images",
        default="/imagelist.txt")
parser.add_option("-o", "--output_for_predictions",
        help="File to store model predictions",
        default="/modelresults.txt")
parser.add_option("-t", "--threshold_value",
        help="Threshold for model confidence value",
        default=0.5)

(options, args) = parser.parse_args()
skip_image=options.skip_image
path_to_images=options.path_to_images
output_for_predictions=options.output_for_predictions
threshold_value=options.threshold_value

# Options
model_file = "/FILEPATH/tortoisemodel.keras"

# Load model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))
model.to(device)
print(f"Model loaded from {model_file}")


# Load in text file, convert to list
img_string = open(path_to_images, 'r')
img_read = img_string.read() 
img_files = img_read.split("\n") 

n_samples = 1
for img_file in img_files:

	imgs = [np.array(Image.open(img_file).convert("RGB"))]

	# Prepare for Tensorflow
	custom_transforms = []
	custom_transforms.append(torchvision.transforms.ToTensor())
	transforms = torchvision.transforms.Compose(custom_transforms)
	imgs = np.array([transforms(img) for img in imgs])
	imgs = torch.from_numpy(imgs)
	imgs.to(device)

	# Get predictions
	model.eval()
	with torch.no_grad():
		all_predictions = model(imgs)

	# Plot predictions

	for i in range(n_samples):
		img = imgs[i]
		predictions = all_predictions[i]
		boxes = predictions.get('boxes', torch.tensor([]))
		scores = predictions.get('scores', torch.tensor([]))

		img = np.swapaxes(img, 0, 2)
		img = np.swapaxes(img, 0, 1)
		
		if not skip_image:
			fig, axs = plt.subplots(1)
			axs.imshow(img)
		
		print("Image: {}".format(i+1))	
		print(len(predictions))
	
		for pi, (bbox, score) in enumerate(zip(boxes, scores)):
			f = open(output_for_predictions, "a")
			if score > threshold_value:
				print(img_file, file=f)
				print(f"  Prediction {pi + 1}: bbox = {bbox}, score = {score}", file=f)
			f.close()
		
			if not skip_image:
				# Convert bbox to integers for plotting
				bbox_int = bbox.int().tolist()
				color = colormaps.get_cmap('RdYlGn')(score)
		
				# Add bounding boxes to plot
				rect = patches.Rectangle((bbox_int[0], bbox_int[1]), bbox_int[2] - bbox_int[0], bbox_int[3] - bbox_int[1], linewidth=1, edgecolor=color, facecolor='none')
				axs.add_patch(rect)
				axs.text(bbox_int[0], bbox_int[1], f'{score:.2f}', fontsize=5, color='white', bbox=dict(facecolor=color, alpha=0.5))
		
		if not skip_image:
			plt.show()

#Predictions set to save above
print("Predictions Completed")
