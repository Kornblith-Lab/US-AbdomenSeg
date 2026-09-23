# Model Investigation Details

Please see full manuscript at [ Link Pending ] [Citation] 

**Objective**: Describe the development and implementation of machine learning models to automatically localize and segment six abdominal organs in pediatric FAST. 

**Study Design**: 
We developed or fine-tuned models based on the U-Net, EVIT U-Net, UltraSAM and nnUnet architectures to automate the segmentation of six key organs in FAST: kidney, bladder, liver, diaphragm, spleen, and spine. 

**Evaluation, Metrics**:
- Accuracy of the automated segmentation was assessed by measuring the Dice-Sørensen similarity (DSC) against manual annotations.
- Segmentation Failures, Mask Acceptability were assessed in the unlabeled datasets by ultrasound experts.

**Study Data Sources**: 
Ultrasound studies were obtained from routine clinical care at multiple institutions, including a Level I pediatric trauma center. FAST examinations were performed at the point of care by treating clinicians and subsequently archived through institutional ultrasound quality assurance programs. All images were re-assigned de-identified ID's and all patient identifiers were removed from DICOM images and metadata 

- a labeled dataset of 200 FAST video clips (“FAST 200”) containing manually annotated frames for six abdominal organs (*Model Training and Testing*)
- an unlabeled pediatric dataset (n = 120 clips; 40 right upper quadrant, 40 left upper quadrant, and 40 suprapubic) (*Mask Evaluation*)
- an unlabeled adult dataset (n = 120 clips; same distribution by view as the unlabeled pediatric (*Mask Evaluation*) 

**Image Acquisition**: POCUS Images from FAST protocol. Most typically B-Mode Ultrasound

<img alt="Study Flow" src="/assets/Study_Flow.png">

**Reference Standard**: Manually Labelled Images by Clinical Experts
[ Link - Labelling Instruction ]

**Image Preprocessing**: 
All images and corresponding segmentation masks were stored at 640x480 and resized from the same standardized resolution to model specific input dimensions with a square crop and padding as appropriate. 
Pixel intensities were normalized prior to training the models. Data augmentation techniques were applied variably across all models to improve generalizability and reduce overfitting, and included random horizontal and vertical flips, rotations, and cropping, with specific parameters varying nominally between model implementations.

**Data Partitions**: Images in Training/Testing were split 80-20% at clip level with simple random sampling. Same split applied to all model types. 

**Models**: CNNs, Transformers or Hybrid Approach

**Software Libraries**: PyTorch, tensorflow (see requirements for each implementation separately.

**Initialization of Model Parameters**: 
(UNet) Default weights from ImageNet 
(Evit) Default weight from EVIT; Initialized with EfficentFormer V2 Weights in the encoder and decoder 
(nnUNet) Default weights from nnUNet
(UltraSam) Default weights from UltraSam; Baseline weights are a version of the SAM model fine-tuned on several ultrasound modalities.

**Training & Final Model Selection**:
(Unet) trained up to 40 epochs. Final Checkpoint selected 
(Evit) default parameters trained up to 300 epochs, Checkpoints saved every 25 epochs. Best checkpoint used
(nnUNet) default parameters for 1 model replicate, trained up to 250 epochs. Final Checkpoint selected.
(UltraSam) default parameters trained up to 5000 iterations. Tested on validation dataset at every 50th Best checkpoint selected. 

**Limitations**: 
Models were developed on a limited pediatric dataset without comprehensive validation. No warrants on models beyond use in included/similar datasets 
