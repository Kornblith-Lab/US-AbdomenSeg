This code is based on the paper by Xin Li, Wenhui Zhu, Xuanzhao Dong, Oana M. Dumitrascu, and Yalin Wang.

Code for the paper: [arXiv](https://arxiv.org/abs/2410.15036)

These notes cover running inference with the per-organ checkpoints. Each checkpoint is a single-organ binary segmentation 
model.

---

## Running Inference

### 1. Set Up the Conda environment

Once you have the EViT repository cloned, build the virtual environment using Conda. The `environment.yml` pins GPU-only packages, so edit these if you are on a machine without CUDA.

**Requirements:** Python 3.10.15 and PyTorch 2.0.1

To create an environment, run the following:
`conda env create -f environment.yml -n <env_name>`

Then activate the environment:

`conda activate <env_name>`


## 2. Structure the model

All organ checkpoints use the same config:

```python
from unet import Eff_Unet

net = Eff_Unet(
    layers=[5, 5, 15, 10],
    embed_dims=[40, 80, 192, 384],
    downsamples=[True, True, True, True],
    vit_num=6,
    drop_path_rate=0.1,
    num_classes=1,
    fork_feat=True).cuda()
```

### 3. Download model weights

Download the pretrained weights for your target 
organ(s) from the `evit_unet` folder. The following checkpoints are available:

- bladder.pth
- diaphragm.pth
- kidney.pth
- liver.pth
- spine.pth
- spleen.pth

## 4. Load weights

```python
net.load_state_dict(torch.load("organ.pth"), strict=True)
net.eval()
```
Replace `"organ.pth"` with your organ checkpoint.

## 4. Format an image and produce a prediction

The model expects a single-channel 224×224 float tensor `[1, 1, 224, 224]`.

```python
img = cv2.imread("frame.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
x, y = img.shape
slice_ = zoom(img, (224 / x, 224 / y), order=3)                # order=3 for image
inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().cuda()

with torch.no_grad():
    out = net(inp)
mask = out.cpu().detach().numpy().squeeze() > 0.5              # bool [224, 224]
mask_full = zoom(mask.astype(np.float32), (x / 224, y / 224), order=0) > 0.5
```

The resize/tensor/threshold steps are taken 
directly from `utils.py` and `png_to_npz.py`. The 
`order=3` for images / `order=0` for masks convention 
is from `utils.py`.

To run it: save the above as `single_infer.py` in the 
repo root, edit `"organ.pth"` and `"frame.png"`, then 
`python single_infer.py`.




