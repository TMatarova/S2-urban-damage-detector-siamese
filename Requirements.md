### Dependencies

* GDAL
* Rasterio
* NumPy
* Matplotlib
* PyTorch (+ torchvision if you use any vision utils)
* scikit-learn
* pandas

### requirements.txt (third-party only)

```txt
gdal>=3.6
rasterio>=1.3
numpy>=1.24
matplotlib>=3.7
torch>=2.2
torchvision>=0.17
scikit-learn>=1.3
pandas>=2.0
```

### Recommended install (Conda, easiest for GDAL/Rasterio)

```bash
# Create env (Python 3.11 is a safe pick)
conda create -n s2siamese -c conda-forge python=3.11 gdal rasterio numpy pandas scikit-learn matplotlib
conda activate s2siamese

# Then install PyTorch (CPU):
pip install torch torchvision

# Or PyTorch with CUDA (if you have an NVIDIA GPU + CUDA toolchain):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

### Pip-only (if you must)

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Drop this straight into your README. If you also rely on YAML configs, add `pyyaml` to the requirements.
