# Adversarial Learning-based Domain Adaptation Algorithm for Intracranial Artery Stenosis Detection on Multi-source datasets
#### *by: Chenbin Ma, Lishuang Guo, Xunming Ji*


## Supplementary Material and code for **ALDA**

<p align="center">
<img src="misc/fig1.tif" width="900" class="center">
</p>

## High-quality Figures
please check [File](https://github.com/chenbinma/ALDA/tree/main/misc).

## Datasets
### Available Datasets
We used our collected private Intracranial Artery Stenosis (ICAS) datasets and public diabetic retinopathy (DR) datasets in this study.
- [ICAS dataset]() 
  - [ICAS-A](https://github.com/chenbinma/ALDA/tree/main/datasets/ICAS-A)
  - [ICAS-B](https://github.com/chenbinma/ALDA/tree/main/datasets/ICAS-B)
  - [ICAS-C](https://github.com/chenbinma/ALDA/tree/main/datasets/ICAS-C)
  - [ICAS-D](https://github.com/chenbinma/ALDA/tree/main/datasets/ICAS-D)
- [DR dataset]() 
  - [EyePACS](http://www.eyepacs.com/data-analysis)
  - [DDR](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset)
  - [ODIR](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
  - [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

## Implementation Code
### Requirmenets:
- Python3
- Pytorch==1.0.1
- Numpy==1.16.2
- scikit-learn==0.20.3
- Pandas==0.23.4
- Matplotlib==3.0.2
- skorch==0.9.0 (For DEV risk calculations)
- openpyxl==2.5.8 (for classification reports)
- Wandb=0.8.28 (for sweeps)

### Adding New Dataset
#### Structure of data
To add new dataset (*e.g.,* NewData), it should be placed in a folder named: NewData in the datasets directory.

#### Configurations
Next, you have to add a class with the name NewData in the `configs/data_model_configs.py` file. 
You can find similar classes for existing datasets as guidelines. 
Also, you have to specify the cross-domain scenarios in `self.scenarios` variable.

Last, you have to add another class with the name NewData in the `configs/hparams.py` file to specify
the training parameters.


### Existing Algorithms
- [DRIT++](https://github.com/HsinYingLee/DRIT)
- [CycleGAN](https://github.com/junyanz/CycleGAN)
- [ITTR](https://github.com/lucidrains/ITTR-pytorch)
- [DCAC](https://github.com/ShishuaiHu/DCAC)
- [DoFE](https://github.com/emma-sjwang/Dofe)
- [ALDA](https://github.com/chenbinma/ALDA)


### Adding New Algorithm
To add a new algorithm, place it in `algorithms/algorithms.py` file.

## Training procedure

The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by `--experiment_description`.
- Each experiment could have different trials, each is specified by `--run_description`.
- For example, if we want to experiment different *ALDA Detection* methods with CNN backbone, we can assign
`--experiment_description CNN_backnones --run_description TDModel` and `--experiment_description CNN_backnones --run_description ALDA` and so on.


## Results
- Each run will have all the cross-domain scenarios results in the format `src_to_trg_run_x`, where `x`
is the run_id (you can have multiple runs by assigning `--num_runs` arg). 
- Under each directory, you will find the classification report, a log file, checkpoint, 
and the different risks scores.
- By the end of the all the runs, you will find the overall average and std results in the run directory.


## Citation
If you found this work useful for you, please consider citing it.
```
@article{ALDA,
  title   = {Adversarial Learning-based Domain Adaptation Algorithm for Intracranial Artery Stenosis Detection on Multi-source datasets},
  author  = {Yuan Gao, Chenbin Ma, Lishuang Guo, Xuxiang Zhang, Xunming Ji},
  journal = {####},
  year    = {2023}
}
```

## Contact
For any issues/questions regarding the paper or reproducing the results, please contact any of the following.   

Chenbin Ma:  *machenbin@buaa.edu.cn*

Xunming Ji:   *jixm@ccmu.edu.cn*   

Department of Biomedical Engineering, Beihang University, 
37 Xueyuan Road, Beijing, 100853

## Acknowledgement
We would like to thank the following repositories for their valuable contributions to this work:
- [DRIT++](https://github.com/HsinYingLee/DRIT)
- [CycleGAN](https://github.com/junyanz/CycleGAN)
- [ITTR](https://github.com/lucidrains/ITTR-pytorch)
- [DCAC](https://github.com/ShishuaiHu/DCAC)
- [DoFE](https://github.com/emma-sjwang/Dofe)