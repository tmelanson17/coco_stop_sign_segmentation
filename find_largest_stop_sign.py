from dataloader import CocoDataloader
import cv2
import numpy as np


# Initialize dataloader and category ID for 'stop sign'
dl = CocoDataloader()
image_ids, category_id = dl.get_image_category("stop sign")
image_iterator = dl.images(image_ids)

largest_mask_size = 0
largest_mask = None
mask_image = None
for image, image_data in image_iterator:
    anns = dl.get_annotations(category_id)
    for ann in anns:
        segmentation = ann['segmentation']
        mask = dl.ann_to_mask(ann)
        if np.sum(mask) > largest_mask_size:
            largest_mask = mask
            largest_mask_size = np.sum(mask)
            mask_image = image


print(f"Pixel size of largest mask: {largest_mask_size}")
cv2.imshow("Largest stop sign", cv2.bitwise_and(mask_image, mask_image, mask=largest_mask))
cv2.waitKey(0)
