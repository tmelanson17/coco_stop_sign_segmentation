from pycocotools.coco import COCO
from config import Files

import cv2
import os


class CocoDataIterator:
    def __init__(self, files, image_ids, coco):
        self.fl = files
        self.image_ids = image_ids
        self.coco = coco

    def __iter__(self):
        self.image_ids_iter = iter(self.image_ids)
        return self

    def __next__(self):
        try:
            self.curr_image_id = next(self.image_ids_iter)
        except StopIteration as e:
            raise e
        image_data = self.coco.loadImgs(self.curr_image_id)[0]
        image_path = os.path.join(self.fl.dataDir, f'{self.fl.dataType}/{image_data["file_name"]}')
        image = cv2.imread(image_path)
        return image, image_data

    def get_current_annotations(self, category_id):
        # Get annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=[self.curr_image_id], catIds=[category_id])
        anns = self.coco.loadAnns(ann_ids)
        return anns

class CocoDataloader:
    def __init__(self):
        self.fl = Files()
        # Create a directory to save the segmented stop sign images
        os.makedirs(self.fl.output_dir, exist_ok=True)
        self.coco = COCO(self.fl.annFile)

    def images(self, image_ids):
        self.iter = CocoDataIterator(self.fl, image_ids, self.coco)
        return iter(self.iter)

    def write_image(self, output_filename, image_data):
        # Save the segmented stop sign as a separate image
        output_filename = os.path.join(self.fl.output_dir, output_filename)
        cv2.imwrite(output_filename, image_data)
         
    def get_image_category(self, name):
        category = self.coco.getCatIds(catNms=[name])[0]
        return self.coco.getImgIds(catIds=[category]), category

    def get_annotations(self, category_id):
        return self.iter.get_current_annotations(category_id)

    def ann_to_mask(self, ann):
        return self.coco.annToMask(ann)
