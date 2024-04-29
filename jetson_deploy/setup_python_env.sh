# wget https://nvidia.box.com/shared/static/0h6tk4msrl9xz3evft9t0mpwwwkw7a32.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
mamba env create -f jetson_environment.yaml
sudo apt-get install libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev
cp /usr/lib/aarch64-linux-gnu/gstreamer-1.0/* /home/yihuai/miniforge3/envs/umi-jetson/lib/gstreamer-1.0