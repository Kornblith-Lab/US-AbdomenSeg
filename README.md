# US-AbdomenSeg
Research Repository for multi-organ segmentation in abdominal ultrasound using deep learning models 



## Model implementation features

| Model | Source | Implementation | Input image Size (px) | Approximate Model Size on Disk | Mean Inference Time* |
| --- | --- | --- | ---: | ---: | ---: |
| UNet | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | Tensorflow (Trained up to 40 Epochs) | 256 × 256 | 25 MB | 35 ms |
| EVIT-UNet | [Retinal-Research/EVIT-UNET](https://github.com/Retinal-Research/EVIT-UNET) | Pytorch (Trained up to 300 Epochs) | 224 × 224 | 250 MB | 54 ms |
| UltraSam | [CAMMA-public/UltraSam](https://github.com/CAMMA-public/UltraSam) | Pytorch (Trained up to 2000 Epochs) | 1024 × 768 (padded) | 360 MB | 124 ms |
| nnUNet | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | Pytorch (Trained up to 250 Epochs) | 960 × 720 | 350 MB | 172 ms |