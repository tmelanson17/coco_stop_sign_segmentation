from dataloader import CocoDataloader
import cv2
import numpy as np

def sort_by_percentage_red(segmentation, threshold=0.8):
    LOW_THRESHOLD=100
    HIGH_THRESHOLD=200
    mask = segmentation[:,:,0] < LOW_THRESHOLD
    mask = np.bitwise_and(mask, segmentation[:,:,1] < LOW_THRESHOLD)
    mask = np.bitwise_and(mask, segmentation[:,:,2] > HIGH_THRESHOLD)
    return np.sum(mask) / np.sum(segmentation[:,:,2] > 0) > threshold

# Sort by at least x% of the image
def sort_by_size(segmentation, threshold=0.1):
    mask = segmentation[:,:,0] > 0
    h,w,_ = segmentation.shape
    return np.sum(mask) / (h*w) > threshold




# Initialize dataloader and category ID for 'stop sign'
dl = CocoDataloader()
image_ids, category_id = dl.get_image_category("stop sign")
image_iterator = dl.images(image_ids)
total_count=0
threshold_count=0

for image, image_data in image_iterator:
    anns = dl.get_annotations(category_id)
    for ann in anns:
        segmentation = ann['segmentation']
        mask = dl.ann_to_mask(ann)

        # Apply the mask to the image to segment the stop sign
        segmented_stop_sign = cv2.bitwise_and(image, image, mask=mask)
        total_count+=1
        if sort_by_size(segmented_stop_sign) and sort_by_percentage_red(segmented_stop_sign, 0.2):
            threshold_count+=1

            # Save the segmented stop sign as a separate image
            output_filename = f'stop_sign_{image_data["file_name"]}'
            dl.write_image(output_filename, segmented_stop_sign)

print(f"{threshold_count}/{total_count}")
print(f"Saved files in {dl.fl.output_dir}")
