# DepthTransfer (coming soon)
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
