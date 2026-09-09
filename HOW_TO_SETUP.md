# 🛠️ How to Setup: Quick Reference Guide

> [!NOTE]
> For the complete, detailed installation manual with troubleshooting steps, hardware requirements, and FAQ, please refer to [SETUP.md](file:///c:/Users/kampa/OneDrive/Desktop/ReportTest/SETUP.md).

---

## ⚡ Quick 5-Minute Setup

### 1. Prerequisites
- **Windows 10/11 (64-bit)**
- **NVIDIA GPU** with CUDA support (e.g. RTX 3060 / 4050 / 4060, minimum 6GB VRAM)
- **Anaconda / Miniconda** installed
- **Ollama** installed

### 2. Pull Ollama Model
```powershell
ollama pull qwen2.5-coder:3b
```

### 3. Setup Python Conda Environment
```powershell
conda create -n paddle_vl python=3.11 -y
conda activate paddle_vl
```

### 4. Install GPU PaddlePaddle
For CUDA 11.8:
```powershell
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```
For CUDA 12.x:
```powershell
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
```

### 5. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 6. Run Automated Test Suite (114 Tests)
```powershell
python -m pytest tests/ -v
```

### 7. Launch Streamlit Web UI
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

See [SETUP.md](file:///c:/Users/kampa/OneDrive/Desktop/ReportTest/SETUP.md) for full configuration details, section batch workflows, and troubleshooting tips.
