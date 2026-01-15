<div align="center">

## [Seeing Through the Rain: Resolving High-Frequency Conflicts in Deraining and Super-Resolution via Diffusion Guidance](https://arxiv.org/pdf/2511.12419)

<div>
    <a href='https://24wenjie-li.github.io/' target='_blank'>Wenjie Li</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=9tYW9LcAAAAJ&hl=zh-CN&oi=ao' target='_blank'>Jinglei Shi</a><sup>2</sup>&emsp;
    <a href='https://hjynwa.github.io/' target='_blank'>Jin Han</a><sup>3</sup>&emsp;
    <a href='https://gh-home.github.io/' target='_blank'>Heng Guo</a><sup>1</sup>&emsp;
    <a href='https://zhanyuma.cn/index.html' target='_blank'>Zhanyu Ma</a><sup>1</sup>
</div>
<div>
    <sup>1</sup>Beijing University of Posts and Telecommunications&emsp; 
    <sup>2</sup>Nankai University&emsp; 
    <sup>3</sup>Noah’s Ark Lab&emsp; 
</div>

<img src="assets/network.png" width="800px"/>

:star: If DHGM is helpful to your images or projects, please help star this repo. Thanks! :hugs: 

---
</div>


### Dependencies and Installation

- Pytorch >= 1.8.1
- CUDA >= 11.1
- basicsr 1.4.2

```
# For install basicsr
pip install basicsr==1.4.2

python setup.py develop -i http://mirrors.aliyun.com/pypi/simple/

python -m pip install --upgrade pip

pip install numpy==1.24.4

pip install -v -e .
```

### Quick Start

#### Prepare Training Data: 
Download our processed training data from [[Google Drive]()] to the input data folder. (Coming soon)


#### Prepare Testing Data:
Download our processed testing data from [[Google Drive]()] to the input data folder. (Coming soon)

#### Visual Results
You can download the qualitative results of our DHGM from [[Google Drive](https://drive.google.com/file/d/17YOUJEYmlWsKX99-qkJKSi5bkp5pJRDJ/view?usp=drive_link)].


### Train

```
# For Stage I
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train_OursS1_x2_syn.yml --launcher pytorch

# For Stage II
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train_OursS2_x2_syn.yml --launcher pytorch
```

### Test

#### Download Pre-trained Models:
Download the pretrained models from [[Google Drive](https://drive.google.com/file/d/1syTqnQs7Uk9JxNBf_FbZCQhlsGW5DT_z/view?usp=drive_link)] to the `experiments/Ours/models` folder. 


```
python basicsr/test.py -opt options/test_Ours_x2_syn.yml
```

### Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

    @inproceedings{li2026seeing,
      title={Seeing Through the Rain: Resolving High-Frequency Conflicts in Deraining and Super-Resolution via Diffusion Guidance},
      author={Li, Wenjie and Shi, Jinglei and Han, Jin and Guo, Heng and Ma, Zhanyu},
      booktitle={AAAI},
      year={2026}
    }


### Acknowledgement
The foundation for the training process is [BasicSR](https://github.com/XPixelGroup/BasicSR), which profited from the outstanding contribution of [XPixelGroup](https://github.com/XPixelGroup) .

### Contact
This repo is currently maintained by lewj2408@gmail.com and is for academic research use only. 
