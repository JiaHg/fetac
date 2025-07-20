# [EAAI 2025] Feature transformation and statistical calibration for cross-domain few-shot classification

This is the implementation of our EAAI 2025 paper: Feature transformation and statistical calibration for cross-domain few-shot classification.

## Datasets

**Source Domain**: NWPU45

**Target Domains**: AID, PatternNet, RSI-CB128, GID, Place25
Please download the datasets from the official websites.

## Data Preparation

After downloading the datasets, remove overlapping classes following Table A.6 in our paper to ensure proper cross-domain few-shot learning setup.

## Meta-Training

We provide pre-trained backbones trained on the NWPU45 source domain for ResNet-18 backbone and Conv-4 backbone. Download the models from the `models/` folder or train your own backbones.

## Meta-Testing

For FETAC with Conv-4 backbone on AID dataset (1-shot):

```python
python test_target.py \
    --data.dataset_dir /home/fetac/data/l_AID \
    --data.shot_num 1 \
    --data.total_file 4420 \
    --data.test_start 13 \
    --test.use_lightFiLM True \
    --test.use_rsa True \
    --test.rectify lla \
    --model.name conv4 \
    --model.weight_path models/42conv4.pth.tar
```

To run experiments with different settings, modify the parameters in `config.py`. The key parameters are described below:

**data args:**

- `data.dataset_dir` : the path of the test dataset
- `data.shot_num` : the number of shot
- `data.total_file` : the number of the test dataset images
- `data.test_start` : the number of the test dataset classes

**test args:**

- `test.use_lightFiLM` : whether use lightFiLM
- `test.use_rsa` : whether use the rsa
- `test.rectify` : use what types of rectification
  
    **For FETAC:**
    
    ```python
    --test.use_lightFiLM True
    --test.use_rsa True
    --test.rectify lla
    ```
    
    **For TSA+LLA:** 
    
    ```python
    --test.use_lightFiLM False
    --test.use_rsa True
    --test.rectify lla
    ```
    

**model args:**

- `model.name` : use what types of backbones
- `model.weight_path` : the path of pretrained backbone

## **Acknowledgments**

Our code builds upon [URL](https://github.com/VICO-UoE/URL) and [STF](https://github.com/Frankluox/Channel_Importance_FSL) repositories. 

## **Citation**

If you find FETAC useful in your research, please cite:

```
@article{liu2025feature,
  title={Feature transformation and statistical calibration for cross-domain few-shot classification},
  author={Liu, Jiafan and Deng, Jin and Cui, Jinrong and Luo, Wei},
  journal={Engineering Applications of Artificial Intelligence},
  volume={157},
  pages={111181},
  year={2025},
  publisher={Elsevier}
}
```
