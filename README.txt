## 🛠️ Installation & Dependencies

This pipeline requires several system-level C++ libraries to handle raw network packet sniffing, serialization, and mathematical optimization.

**1. Install System Dependencies (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git \
    libzip-dev libpcap-dev libtins-dev \
    flatbuffers-compiler libflatbuffers-dev libceres-dev