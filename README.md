# Learned Proximal Networks for Quantitative Susceptibility Mapping (LPCNN)
[MICCAI](https://www.google.com "Google's Homepage") | [Arxiv](https://www.google.com "Google's Homepage")
<!--- put link here --->

### Official pytorch implementation of the paper<br>
In this repository, we provides our official implementation of LPCNN network and the QSM train dataset for MICCAI.<br>
![alt text](https://github.com/Sulam-Group/LPCNN/blob/master/imgs/overall_framework.png "overall framework")

## Requirements
- [Python 3.6](https://www.python.org/)
- [PyTorch 1.2.0](https://pytorch.org)

## Environment Settings
Users can use the command to install all requried libraries.
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
### Test on validation set 
```
python LPCNN/main.py --mode predict --number [PHASE_INPUT_NUMBER] --tesla [TESLA] --model_arch lpcnn --resume_file [MODEL_CHECKPOINT] --pred_set val --size whole
```
Template command using provided dataset:
```
python LPCNN/main.py --mode predict --number 1 --tesla 7 --model_arch lpcnn --resume_file checkpoints/lpcnn_test_Emodel.pkl --pred_set val --size whole
```
### Test on your own data
```
python LPCNN/inference.py --save_name [SAVE_NAME] --number [PHASE_INPUT_NUMBER] --tesla [TESLA] --phase_file [PHASE_DATA_PATH] --dipole_file [DIPOLE_DATA_PATH] --mask_file [MASK_DATA_PATH] --model_arch lpcnn --resume_file [MODEL_CHECKPOINT] 
```
If you have ground-truth QSM result, you can add --gt_file `[GT_DATA_PATH]` and the function will calculate the performance.<br>
Template command using provided dataset (number=3):
```
python LPCNN/inference.py --number 3 --phase_file test_data/three/phase_data3.txt --dipole_file test_data/three/dipole_data3.txt --mask_file test_data/three/mask_data3.txt --gt_file test_data/three/gt_data3.txt --resume_file checkpoints/lpcnn_test_Bmodel.pkl
```
