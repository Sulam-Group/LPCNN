# Learned Proximal Networks for Quantitative Susceptibility Mapping (LPCNN)
[MICCAI](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_13) | [Arxiv](https://arxiv.org/abs/2008.05024 "Arxiv")
<!--- put link here --->

### Official pytorch implementation of the paper<br>
LPCNN was developed by Kuo-Wei Lai, Dr. Xu Li and Dr. Jeremias Sulam, for solving the ill-posed dipole deconvolution problem in Quantitative Susceptibility Mapping (QSM). By integrating proximal gradient descent with deep learning, it is the first deep learning based QSM method that can handle an arbitrary number of phase input measurements. In this repository, we provides official implementation of LPCNN network and the QSM training datasets (n=8, with local phase data acquired at 7T and 4-5 orientations COSMOS).<br><br>
The PyTorch implementations of LPCNN that offer the following functions:
 - Create default training dataset including patched local phase image and QSM target pairs
 - Conduct single or multiple orientation dipole deconvolution training using LPCNN
 - Reconstruct QSM maps with user’s own data using trained LPCNN model
<br><br>
![alt text](https://github.com/Sulam-Group/LPCNN/blob/master/imgs/overall_framework.png "overall framework")

## Requirements
- [Python 3.6](https://www.python.org/)
- [PyTorch 1.2.0](https://pytorch.org)

## Environment Settings
use the command below to install all requried libraries.
```
conda env create --name [MY_ENV] -f environment.yml
```
## Usage
activate conda environment first
```
conda activate [MY_ENV]
```
### Dataset
We provide the script to generate the dataset we used in our paper. `[NUMBER]` here indicates single or multiple phase input dataset.
```
./create_dataset.sh [NUMBER]
```
### Train
```
python LPCNN/main.py 

arguments:
--mode                        train or predict [default is train]
--name                        name of the experiment and model [default is _test]
--number                      number of the phase input, choices=[1, 2, 3], [default is 1]
--tesla                       tesla of the dataset
--gpu_num                     number of gpus [default is 1]
--model_arch                  network architecture [default is lpcnn]
--num_epoch                   number of total epochs to train [default is 100]
--batch_size                  batch size [default is 2]
--learning_rate               learning rate [default is 1e-4]
--optimizer                   optimizer to use [default is adam]
--momentum                    SGD momentum [default is 0.9]
--no_cuda                     disables CUDA training [default is false]
--resume_file                 resume training from checkpoint [default is None]
```
Template command using provided dataset:
```
python LPCNN/main.py --mode train --name _test --number 1 --tesla 7 --model_arch lpcnn --num_epoch 100 --batch_size 2 --learning_rate 0.0001 --optimizer adam
```
#### Tensorboard Visualization
During training, users can monitor the training by using the following command:
```
tensorboard --port 6006 --logdir LPCNN/tb_log
```
### Test on validation set 
```
python LPCNN/main.py

arguments:
--mode                        train or predict [default is train]
--number                      number of the phase input, choices=[1, 2, 3], [default is 1]
--tesla                       tesla of the dataset
--gpu_num                     number of gpus [default is 1]
--model_arch                  network architecture [default is lpcnn]
--no_cuda                     disables CUDA training [default is false]
--no_save                     disable saving result [default is false]
--resume_file                 resume saved checkpoint for testing [default is None]
--pred_set                    choose which set for testing, choices=[train, val, test], [default is val]
--size                        choose whole or patch images for testing, choices=['whole', 'patch'], [default is whole]
```
Template command using provided dataset:
```
python LPCNN/main.py --mode predict --number 1 --tesla 7 --model_arch lpcnn --resume_file checkpoints/lpcnn_test_Emodel.pkl --pred_set val --size whole
```
### Test on your own data
```
python LPCNN/inference.py

arguments:
--save_name                   save out name
--number                      number of the phase input, choices=[1, 2, 3], [default is 1]
--phase_file                  phase data list path
--dipole_file                 dipole data list path
--mask_file                   mask_data path
--gt_file                     ground-truth data path (optional)
--tesla                       tesla of the dataset
--gpu_num                     number of gpus [default is 1]
--model_arch                  network architecture [default is lpcnn]
--no_cuda                     disables CUDA training [default is false]
--resume_file                 resume saved checkpoint for testing [default is None]
--crop                        crop redundant background margin to reduce memory usage
```
If you have ground-truth QSM result, you can add `--gt_file [GT_DATA_PATH]` and the function will calculate the performance.<br>
Template command using provided dataset (number=3):
```
python LPCNN/inference.py --number 3 --phase_file test_data/three/phase_data3.txt --dipole_file test_data/three/dipole_data3.txt --mask_file test_data/three/mask_data3.txt --gt_file test_data/three/gt_data3.txt --resume_file checkpoints/lpcnn_test_Bmodel.pkl
```
