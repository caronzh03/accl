#! /bin/bash

# Bootstrap scripts for Ubuntu LTS 22.04

# Install utility packages
sudo apt update && sudo apt install vim tmux tree

# Install prepatory packages for CUDA
sudo apt install build-essential
sudo apt install nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc

# Purge previously installed nvidia driver, if any
sudo apt remove --purge nvidia-*
sudo apt autoremove --purge
sudo reboot

# Install nvidia driver 520 (only this version works on Ubuntu 22.04)
sudo apt install nvidia-driver-520

# Verify driver installation is successful
nvidia-smi

# Install CUDA toolkit (compiler)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
sudo reboot

# Post installation ENV variable setup

# Add the following lines to ~/.bashrc and /etc/environment
export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc
source /etc/environment

# Verification
lsmod | grep nvidia # should show nvidia* content
nvidia-smi # should show CUDA version: 12.0
nvcc --version # should show compiler version 12.0  <- this version cannot be higher than smi
gcc --version # 11.3.0
g++ --version # 11.3.0





############## Detailed notes ###############

sudo apt update && sudo apt install vim tmux tree

# Install conda (optional)
cd ~/Downloads
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh

source ~/anaconda3/bin/activate
conda install cuda -c nvidia

# disable activation by default:
conda config --set auto_activate_base false

# end install conda

sudo apt install build-essential
sudo apt install nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc

# if previously installed different versions of nvidia driver, need uninstall first
# uninstall nvidia driver: https://linuxhint.com/install-nvidia-drivers-on-ubuntu/
sudo apt remove --purge nvidia-*
sudo apt autoremove --purge
sudo reboot

# system font size should shrink back to default

# install driver 520
sudo apt install nvidia-driver-520

# install toolkit 12.0 (nvcc)

# Follow this video: https://www.youtube.com/watch?v=nMeDPN5oIcM

# Step 1: go to "CUDA toolkit archive": https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

# Step 2: select deb (local) Installer Type & follow instructions

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

sudo reboot

# Step 3: Post-installation (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
# Add the following 2 lines to both ~/.bashrc and /etc/environment
export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc
source /etc/environment

# nvcc --version should work now

# verification of successful setup

lsmod | grep nvidia # should show nvidia* content
nvidia-smi # should show CUDA version: 12.0
nvcc --version # should show compiler version 12.0  <- this version cannot be higher than smi
gcc --version # 11.3.0
g++ --version # 11.3.0
