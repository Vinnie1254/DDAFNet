## Environment Installation
The required environment can be installed as follows:
```bash
conda create -n torch1.7 python=3.8 -y
conda activate torch1.7

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorboard scikit-image lpips torch-fidelity pytorch-msssim pillow pandas tqdm opencv-python
```


## Datasets 
```
UIEDB
├── train
│   ├── input      # underwater images
│   │   ├── xxxx.png
│   │   └── ......
│   └── target     # reference images
│       ├── xxxx.png
│       └── ......
└── test
    ├── input      # underwater images
    │   ├── xxxx.png
    │   └── ......
    └── target     # reference images
        ├── xxxx.png
        └── ......

```


## Training
```bash
python train_water_LSUI.py
```


## Testing
```bash
python test_water_LSUI.py
```
