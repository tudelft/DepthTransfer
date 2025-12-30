# DepthTransfer (coming soon)
# 1. Introduction
# 2. Installation
Code for article: Depth Transfer: Learning to See Like a Simulator for Real-World Drone Navigation
Run the following command to get the code for depth_transfer：
``` bash
git clone git@github.com:tudelft/DepthTransfer.git
conda env create -f depth_transfer.yaml
```
Then install the IsaacGym, run the following command:
```bash
conda activate depth_transfer
wget -O IsaacGym_Preview_4_Package.tar.gz https://developer.nvidia.com/isaac-gym-preview-4
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python/
pip install -e .
```
After installing IsaacGym, the implementation of AerialGym can be installed:
```bash
conda activate depth_transfer
cd DepthTransfer/
pip install -e .
```
If you want to test domain adaptation for stereo depth in AeriaGym, install the CUDA (12.6) first (If you already have CUDA installed, skip this step):
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
echo "export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
```
Then go to the folder of ```sgm```, build and install sgm algorithm based on CUDA:
```bash
conda activate depth_transfer
cd DepthTransfer/sgm/
mkdir build && cd build
cmake ..
make -j
cp ../setup.py .
pip install .
```
# 3. Training
