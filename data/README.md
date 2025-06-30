# Data & Pre-trained Models

Please download the following datasets and pretrained models, and put them into the specified directory.

## Preparing datasets

* [x] VISION24


## Dataset Structure

Final directory structure (only display used directories and files):

```
./data
├── VISION24
│   ├── SegmentationClassAug
│   ├── JPEGImages
│   ├── weights
│   
└── README.md

```

### VISION24

* Download [Training data paths] in lists/vision24/train.txt 
* Download [Validation data paths] in lists/vision24/test.txt

 Generate from datasets:
  
    ```bash
    # Dry run to ensure the output path are correct.
    cuda 0 python tools.py precompute_loss_weights with dataset=VISION24 dry_run=True
    # Then generate and save to disk.
    cuda 0 python tools.py precompute_loss_weights with dataset=VISION24
    ```
