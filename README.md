# US-AbdomenSeg
Research Repository for multi-organ segmentation in abdominal ultrasound using deep learning models 



## Model implementation features

| Model | Source | Implementation | Input image Size (px) | Approximate Model Size on Disk | Mean Inference Time* |
| --- | --- | --- | ---: | ---: | ---: |
| UNet | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | Tensorflow (Trained up to 40 Epochs) | 256 × 256 | 25 MB | 35 ms |
| EVIT-UNet | [Retinal-Research/EVIT-UNET](https://github.com/Retinal-Research/EVIT-UNET) | Pytorch (Trained up to 300 Epochs) | 224 × 224 | 250 MB | 54 ms |
| UltraSam | [CAMMA-public/UltraSam](https://github.com/CAMMA-public/UltraSam) | Pytorch (Trained up to 2000 Epochs) | 1024 × 768 (padded) | 360 MB | 124 ms |
| nnUNet | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | Pytorch (Trained up to 250 Epochs) | 960 × 720 | 350 MB | 172 ms |


## Organ Segment Dice Similarity on Pediatric FAST test set

 | Organ | Bladder | Kidney | Liver | Spleen | Spine | Diaphragm |
| --- | --- | --- | --- | --- | --- | --- |
| Manually Labeled Images | n=344 | n=306 | n=445 | n=172 | n=273 | n=267 |
| UNet | 0.90 | 0.92 | 0.84 | 0.70 | 0.69 | 0.62 |
| EVIT-UNet | 0.92 | 0.93 | 0.90 | 0.82 | 0.66 | 0.61 |
| UltraSam | 0.91 | 0.91 | 0.90 | 0.81 | 0.70 | 0.66 |
| nnUNet | 0.92 | 0.85 | 0.89 | 0.60 | 0.69 | 0.67 |
