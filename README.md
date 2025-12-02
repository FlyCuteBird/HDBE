# HDBE
The Code of Exploring High-Dimensional Ball Embedding for Image-text Retrieval
# The code is build on PCME (Probabilistic Cross-Modal Embedding (PCME) CVPR 2021) https://github.com/naver-ai/pcme

## Installation

Install dependencies using the following command.

```
pip install cython && pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt")'
```
import nltk
  >>> nltk.download('punkt_tab')
  
## Dataset preparation

### COCO Caption

We followed the same split provided by [VSE++](http://www.cs.toronto.edu/~faghri/vsepp/data.tar).
Dataset splits can be found in [datasets/annotations](datasets/annotations).

Note that we also need `instances_<train | val>2014.json` for computing PMRP score.

### CUB Caption

Download images (CUB-200-2011) from [this link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), and download caption from [reedscot/cvpr2016](https://github.com/reedscot/cvpr2016).
You can use the image path and the caption path separately in the code.

## Evaluate pretrained models


### COCO Caption

# Compute plausible match R-Precision (PMRP) metric
````
# python extract_rankings_coco_fusion.py ./config/coco/pcme_coco.yaml --dataset_root datasets --model_path1 bert_model_last.pth --model_path2 bert_model_last1.pth --dump_to dump_ranking_file 
# python evaluate_pmrp_coco.py --ranking_fname  dump_ranking_file --dataset_root datasets --n_fold True (False) 
````
# CUB Caption
````
python evaluate_cub.py ./config/cub/pcme_cub.yaml \
    --dataset_root datasets \
    --caption_root ./datasets/annotations/cub/cub_captions \
    --model_path1 model_last.pth \
    --model_path2 model_last1.pth
````
## How to train

NOTE: we train each model  on a single A800-80G. For each model, we need train i2t and t2i direction (see probemb.py), and then fuse them. <br>

### COCO Caption

```
python train_coco.py
    
```
### CUB 
````
python train_cub.py 
````
