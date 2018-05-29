# Feature Space Transfer for Data Augmentation

by Bo Liu, Xudong Wang, Mandar Dixit, Roland Kwitt, and Nuno Vasconcelos

This repository is written by Bo Liu at UC San Diego.

## Citation

If you use our code/model/data, please cite our paper:

    @inproceedings{liu18featuretransfer,
      author = {Bo Liu, Xudong Wang, Mandar Dixit, Roland Kwitt, and Nuno Vasconcelos},
      Title = {Feature Space Transfer for Data Augmentation},
      booktitle = {CVPR},
      Year  = {2018}
    }


## Requirements

1. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.

2. Caffe MATLAB wrapper is required to run the experiments. 

3. LibLinear is required for svm experiments.

## Installation

1. Install all requirements.

2. Clone the FeatureTransfer repository, and we'll call the directory that you cloned FeatureTransfer into `Transfer_ROOT`
    ```Shell
    git clone https://github.com/BoLiu-SVCL/FeatureTransfer.git
    ```
  
## Disclaimer

If you encounter any issue when using our code or model, please let me know.
