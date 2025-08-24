# bird
Bird Feeder Project  

<br>
files included in github repository

- results (csv output from python tracking)
- runs (saved data from training/validation)
    - detect
        - train (yolo12n training)
        - train2 (rt-detr training)
        - train3 (yolo12n, augmentation training)
        - val (yolo1n validation)
        - val2 (rt-detr validation)
        - val3 (yolo12n, augmentation validation)
- detection.ipynb (python tracking)
- detr.ipynb (detr training)
- inference.ipynb (cli inference)
- yolo.ipynb (yolo training)

<br>

files ignored for github repository
- all .pt model files
- runs
    - detect
        - predict (inference video output)
        - track (detection video output)
        - track2 (detection video output, no feeder)
- sam-barret-xy (annotation dataset)
- videos (video dataset)
- all .pt model files