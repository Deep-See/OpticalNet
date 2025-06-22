<div align="center">

<h1>OpticalNet🔬: An Optical Imaging Dataset and Benchmark Beyond the Diffraction Limit🔍</h1>

[![Conference](https://img.shields.io/badge/CVPR-2025-blue)](https://cvpr.thecvf.com/)
[![Poster](https://img.shields.io/badge/Poster-34146-green)](https://cvpr.thecvf.com/virtual/2025/poster/34146)
[![Project](https://img.shields.io/badge/Project-Page-red)](https://Deep-See.github.io/OpticalNet)

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/opticalnet-an-optical-imaging-dataset-and/semantic-segmentation-on-opticalnet)](https://paperswithcode.com/sota/semantic-segmentation-on-opticalnet?p=opticalnet-an-optical-imaging-dataset-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/opticalnet-an-optical-imaging-dataset-and/semantic-segmentation-on-opticalnet-1)](https://paperswithcode.com/sota/semantic-segmentation-on-opticalnet-1?p=opticalnet-an-optical-imaging-dataset-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/opticalnet-an-optical-imaging-dataset-and/semantic-segmentation-on-opticalnet-2)](https://paperswithcode.com/sota/semantic-segmentation-on-opticalnet-2?p=opticalnet-an-optical-imaging-dataset-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/opticalnet-an-optical-imaging-dataset-and/semantic-segmentation-on-opticalnet-3)](https://paperswithcode.com/sota/semantic-segmentation-on-opticalnet-3?p=opticalnet-an-optical-imaging-dataset-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/opticalnet-an-optical-imaging-dataset-and/semantic-segmentation-on-opticalnet-4)](https://paperswithcode.com/sota/semantic-segmentation-on-opticalnet-4?p=opticalnet-an-optical-imaging-dataset-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/opticalnet-an-optical-imaging-dataset-and/semantic-segmentation-on-opticalnet-5)](https://paperswithcode.com/sota/semantic-segmentation-on-opticalnet-5?p=opticalnet-an-optical-imaging-dataset-and) -->

<div>
    Benquan Wang&emsp;
    Ruyi An&emsp;
    Jin-Kyu So&emsp;
    Sergei Kurdiumov&emsp;
    Eng Aik Chan&emsp;
    Giorgio Adamo&emsp;
    Yuhan Peng&emsp;
    Yewen Li&emsp;
    Bo An
</div>

<div>
    🎈 <strong>Accepted to CVPR 2025</strong>
</div>

<div>
    <h4 align="center">
        • <a href="https://cvpr.thecvf.com/virtual/2025/poster/34146" target='_blank'>[pdf]</a> •
    </h4>
</div>

<div>
    If you find our project helpful, kindly consider ⭐ this repo. Thanks! 🖐️
</div>

</div>

## 📮 News
- Mar. 2025: We are working on presenting our final simulation and training code and datasets, and will release them in a very due time. Stay tuned!
- Feb. 2025: Our paper has been accepted to CVPR 2025 🎉

## 💽 Dataset

The dataset is hosted on Hugging Face Datasets. You can download the dataset by running the following command:

```python
from datasets import load_dataset

dataset = load_dataset("Deep-See/OpticalNet")
```

## 🛠️ Installation for training
### Codes and Environment
```bash
# clone this repository
git clone https://github.com/Deep-See/OpticalNet.git
cd OpticalNet

# create a new anaconda environment
conda create -n opticalnet python=3.9 -y
conda activate opticalnet

# install python dependencies
conda install -y -c pytorch pytorch torchvision torchaudio cudatoolkit=12.1
pip install -r requirements.txt
pip install --editable .
```

Ensure that the CUDA version used by `torch` corresponds to the one on the device.

### 🏃Training
The below commands will train a model on simulation and experimental datasets. The trained models will be saved in the `./checkpoints` directory.

#### Simulation Dataset
```bash
python3 main.py --dir_path <PATH_TO_SIMULATION_DATASET> --model_type <MODEL_NAME>
```

#### Experimental Dataset
```bash
python3 main.py --dir_path <PATH_TO_EXPERIMENTAL_DATASET> --model_type <MODEL_NAME>
```

## 💡 Simluation
In order to perform simulation, an separate environment is required. The simulation environment can be installed by running the following commands:
```bash
# clone this repository
git clone https://github.com/Deep-See/OpticalNet.git
cd OpticalNet

# create a new anaconda environment
conda create -n optical-sim python=3.9 -y
conda activate optical-sim

# install python dependencies
pip install -r simulation_requirements.txt
pip install --editable .
```


## 🔝 Citation
If you find our work useful for your research, kindly consider citing our paper:
```bibtex
@inproceedings{opticalnet,
    title={{OpticalNet}: An Optical Imaging Dataset and Benchmark Beyond the Diffraction Limit},
    author={Wang, Benquan and An, Ruyi, and So, Jin-Kyu and Kurdiumov, Sergei and Chan, Eng Aik and Adamo, Giorgio and Peng, Yuhan and Li, Yewen and An, Bo},
    booktitle={CVPR},
    year={2025}
}
```

