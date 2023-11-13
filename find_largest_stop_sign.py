from dataloader import CocoDataloader
import cv2
import numpy as np


# Initialize dataloader and category ID for 'stop sign'
dl = CocoDataloader()
image_ids, category_id = dl.get_image_category("stop sign")
image_iterator = dl.images(image_ids)

# Tuple of (size, mask, image)
largest_masks = list()
for image, image_data in image_iterator:
    # TODO: integrate with image iterator, not dl
    anns = dl.get_annotations(category_id)
    for ann in anns:
        segmentation = ann['segmentation']
        mask = dl.ann_to_mask(ann)
        largest_masks.append(tuple((np.sum(mask), mask, image, image_data["file_name"])))

largest_masks.sort(
        reverse=True,
        key=lambda data: data[0]
)


for i in range(5):
    size, mask, image, filename = largest_masks[i]
    print(f"Image {filename}")
    print(f"Pixel size of largest mask: {size}")
    print(f"Shapes: {mask.shape} {image.shape}")
    cv2.imshow("Largest stop sign", cv2.bitwise_and(image, image, mask=mask))
    cv2.waitKey(0)
