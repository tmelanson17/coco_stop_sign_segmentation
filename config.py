import os
from dataclasses import dataclass

# Specify the paths to your COCO dataset annotations and images
@dataclass
class Files:
  dataDir:  str = '/home/tmelanson/data/coco2017'  # Change this to your COCO dataset directory
  dataType:  str = 'train2017'  # You can use 'val2017' or 'test2017' too
  output_dir: str = '/home/tmelanson/output/stopsigns'
  annFile:  str = os.path.join(dataDir, f'annotations/instances_{dataType}.json')

if __name__ == "__main__":
    f = Files()
