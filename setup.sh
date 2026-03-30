set -e 

echo "=== 1/3: Installing APT Dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    libzip-dev libpcap-dev libtins-dev \
    flatbuffers-compiler libflatbuffers-dev libceres-dev

echo "=== 2/3: Initializing Git Submodules ==="
git config --global url."git@github.com:".insteadOf "https://github.com/"
git config submodule.third_party/ouster_sdk/sdk-extensions.update none
git submodule update --init --recursive

echo "=== 3/3: Setup Complete! ==="
echo "You are ready to compile. Run:"
echo "mkdir build && cd build && cmake .. && make -j\$(nproc)"