from dataloader import CocoDataloader
import cv2


# Initialize dataloader and category ID for 'stop sign'
dl = CocoDataloader()
image_ids, category_id = dl.get_image_category("stop sign")
image_iterator = dl.images(image_ids)

for image, image_data in image_iterator:
    anns = dl.get_annotations(category_id)
    for ann in anns:
        segmentation = ann['segmentation']
        mask = dl.ann_to_mask(ann)

        # Apply the mask to the image to segment the stop sign
        segmented_stop_sign = cv2.bitwise_and(image, image, mask=mask)

        # Save the segmented stop sign as a separate image
        output_filename = f'stop_sign_{image_data["file_name"]}'
        dl.write_image(output_filename, segmented_stop_sign)

print(f"Saved files in {dl.fl.output_dir}")
