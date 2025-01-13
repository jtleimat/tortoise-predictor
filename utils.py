import numpy as np
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import cv2



class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        num_objs = len(coco_annotation)

        masks = []
        labels = []
        boxes = []

        for i in range(num_objs):
            # Convert segmentation polygons to binary masks
            segm = coco_annotation[i]["segmentation"]
            mask = self._polygon_to_mask(segm, img.size)
            masks.append(mask)
            labels.append(coco_annotation[i]["category_id"])
            
            # Extract and convert bounding boxes
            bbox = coco_annotation[i]["bbox"]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            bbox = [x_min, y_min, x_max, y_max]

            # Debugging info
            if width <= 0 or height <= 0:
                print(f"Skipping bbox dimensions: {bbox}")
                continue
            
            boxes.append(bbox)

        # Handle blank images
        if num_objs == 0:
            masks = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # Dummy bbox
            labels = torch.zeros((0,), dtype=torch.int64)  # Dummy label
        else:
            masks = torch.stack([torch.as_tensor(mask) for mask in masks], dim=0)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        # Debugging info
        #print(f"Boxes: {boxes}")

        img_id = torch.tensor([img_id])

        my_annotation = {
            "labels": labels,
            "masks": masks,
            "boxes": boxes,
            "image_id": img_id,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def _polygon_to_mask(self, polygon, img_size):
        # Convert COCO segmentation polygon to a binary mask
        img_w, img_h = img_size
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        polygon = np.array(polygon).reshape(-1, 2)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return torch.as_tensor(mask, dtype=torch.uint8)

    def __len__(self):
        return len(self.ids)

# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained Mask R-CNN model
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Update the mask predictor to match the number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model
    
def evaluate_model(model, data_loader, device, coco_annotation_path):
    model.eval()
    
    # Collect predictions and ground truth annotations
    all_predictions = []
    all_annotations = []
    all_scores = []
    all_labels = []

    import matplotlib.pyplot as plt
    with torch.no_grad():
        for imgs, annotations in data_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # Show tortoise mask
            #plt.imshow(annotations[0]["masks"][0])
            #plt.show()
            
            # Forward pass
            outputs = model(imgs)
            
            # Process predictions
            for i, output in enumerate(outputs):
                img_id = annotations[i]['image_id'].item()
                print(img_id)
                pred = output
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_masks = pred['masks'].cpu().numpy()
                
                ##can uncomment below to see what masks look like
                #fig, axs = plt.subplots(1,2)
                #axs[0].imshow(annotations[0]["masks"][0])
                #axs[1].imshow(pred_masks[0,0])
                #plt.show()
                
                # Format predictions for COCO evaluation
                for j in range(len(pred_boxes)):
                    all_predictions.append({
                        'image_id': img_id,
                        'category_id': int(pred_labels[j]),
                        'bbox': list(pred_boxes[j]),
                        'score': float(pred_scores[j]),
                        'mask': np.array(pred_masks[j])
                    })
                    all_scores.append(pred_scores[j])
                    all_labels.append(pred_labels[j])

            # Collect ground truth annotations
            for ann in annotations:
                img_id = ann['image_id'].item()
                gt_boxes = ann['boxes'].cpu().numpy()
                gt_labels = ann['labels'].cpu().numpy()
                gt_masks = ann['masks'].cpu().numpy()
                for i in range(len(gt_boxes)):
                    all_annotations.append({
                        'image_id': img_id,
                        'category_id': int(gt_labels[i]),
                        'bbox': list(gt_boxes[i])
                    })
    
    # Evaluate predictions using COCOeval
    coco = COCO(coco_annotation_path)
    coco_dt = coco.loadRes(all_predictions)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    return all_predictions, all_annotations, coco_eval
    
def intersection_over_union(bbox_gt, bbox_pred):

	if isinstance(bbox_gt, dict):
		bbox_gt = [bbox_gt.get('x_min', 0), bbox_gt.get('y_min', 0), bbox_gt.get('x_max', 0), bbox_gt.get('y_max', 0)]
		
	if isinstance(bbox_pred, dict):
		bbox_pred = [bbox_pred.get('x_min', 0), bbox_pred.get('y_min', 0), bbox_pred.get('x_max', 0), bbox_pred.get('y_max', 0)]

	x_max = max(bbox_gt[0], bbox_pred[0])
	x_min = min(bbox_gt[2], bbox_pred[2])
	y_max = max(bbox_gt[1], bbox_pred[1])
	y_min = min(bbox_gt[3], bbox_pred[3])
	
	##calculate area of intersection
	intersect = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)
	
	##calculate area of each
	bboxarea_gt = (bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1)
	bboxarea_pred = (bbox_pred[2] - bbox_pred[0] + 1) * (bbox_pred[3] - bbox_pred[1] + 1)
	
	##compute the intersection over union
	
	iou = intersect/ float(bboxarea_gt + bboxarea_pred - intersect)
	
	return iou