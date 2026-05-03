@echo off
echo Setting up EDU environment for Offroad Segmentation...
conda create -y -n EDU python=3.10
call conda activate EDU
pip install -r requirements.txt
echo Environment setup complete. Use 'conda activate EDU' to begin.
