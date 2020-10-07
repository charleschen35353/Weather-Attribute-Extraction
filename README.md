# Weather Attribute Extraction



## Overview
This project aims to extract weather attributes from a given image.

## Getting started
Please download weights from here: https://drive.google.com/drive/folders/1pgHcN1kikHKete80SqmyY2p9cIL3c67t

And test data: https://drive.google.com/drive/folders/1MIxsMe42t6Grb07g6jAmt5SnIhpe4MDR

Run
python3 infer_attr.py ./path-to-images

## Weather environment prediction model performance

Models aim to achieve high accruacy with small models sizes.

### Sky segmentation
Mask pixel error rate: 1.5% 

Data: ADE20K

### Shadow segmentation
Mask pixel error rate: 10% 

Data: SWIMSEG

### Cloud segmentation
Mask pixel error rate: 4.6% 

Data: SBU-shadow

### Rainy detection
Accruacy: 93%

Data: PRIVATE

### Sunny detection
Accruacy: 93%

Data: PRIVATE

### Sunlight intensity estimation
Mask pixel error rate: 3.1%

Intensity Accracy: 88%

## Other source
This program is also migrated to C++ for software integration. If you are interested in the C++ code, please email me through lc3533@columbia.edu.

## License and Citation
Copyrights reserved by Liang-yu (Charles) Chen

Should you have any enquiries about the project and dataset, please email lc3533@columbia.edu


