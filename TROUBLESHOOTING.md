# Troubleshooting Guide

## Common Issues and Solutions

### 🔴 Installation Issues

#### Issue: `pip install` fails
**Solution:**
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# If specific package fails, install individually
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit matplotlib seaborn scikit-learn pandas tqdm tensorboard
```

#### Issue: CUDA/GPU not detected
**Solution:**
```powershell
# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 🔴 Dataset Issues

#### Issue: "No images found in Dataset"
**Solution:**
1. Verify folder structure:
   ```
   Dataset/
   ├── Training/
   │   ├── Early_Blight/
   │   ├── Healthy/
   │   └── Late_Blight/
   ├── Validation/
   └── Testing/
   ```

2. Check image formats (should be .jpg, .jpeg, or .png)

3. Run dataset check:
   ```powershell
   python data.py
   ```

#### Issue: "Directory does not exist"
**Solution:**
- Edit `config.py` and update paths:
  ```python
  BASE_DIR = Path(__file__).parent
  DATASET_DIR = BASE_DIR / "Dataset"
  ```

### 🔴 Memory Issues

#### Issue: CUDA Out of Memory
**Solutions:**
```powershell
# Option 1: Reduce batch size
python main.py --mode train --batch-size 16

# Option 2: Reduce number of workers
python main.py --mode train --batch-size 32 --num-workers 2

# Option 3: Use CPU (slower)
# Edit config.py:
DEVICE = torch.device('cpu')
```

#### Issue: System RAM exhausted
**Solutions:**
- Reduce `num_workers` in config.py or command line
- Close other applications
- Process dataset in smaller batches

### 🔴 Training Issues

#### Issue: Loss is NaN
**Solutions:**
1. Reduce learning rate:
   ```powershell
   python main.py --learning-rate 0.0001
   ```

2. Check data normalization in `data.py`

3. Verify images are loading correctly:
   ```powershell
   python data.py
   ```

#### Issue: Accuracy not improving
**Solutions:**
1. Train for more epochs:
   ```powershell
   python main.py --epochs 100
   ```

2. Try different learning rate:
   ```powershell
   python main.py --learning-rate 0.0005
   ```

3. Check class imbalance:
   ```powershell
   python data.py
   ```

4. Increase data augmentation in `config.py`

#### Issue: Training is very slow
**Solutions:**
1. Reduce batch size affects speed and memory differently:
   ```powershell
   python main.py --batch-size 64  # Faster but needs more memory
   ```

2. Use GPU instead of CPU:
   - Install CUDA-enabled PyTorch
   - Check `config.DEVICE`

3. Increase `num_workers`:
   ```powershell
   python main.py --num-workers 8
   ```

### 🔴 Evaluation Issues

#### Issue: "Checkpoint not found"
**Solution:**
1. Train model first:
   ```powershell
   python main.py --mode train --model cnn --epochs 5
   ```

2. Check checkpoint directory:
   ```powershell
   dir checkpoints
   ```

3. Verify checkpoint path in code matches actual file location

#### Issue: Poor evaluation metrics
**Solutions:**
1. Train for more epochs
2. Use transfer learning (ResNet50):
   ```powershell
   python main.py --mode train --model resnet --epochs 50
   ```
3. Check for data leakage between train/val/test sets
4. Verify data augmentation is only applied to training data

### 🔴 Streamlit App Issues

#### Issue: "Model not found" in Streamlit
**Solution:**
1. Train and save model first:
   ```powershell
   python main.py --mode train --model resnet --epochs 10
   ```

2. Verify checkpoint exists:
   ```powershell
   dir checkpoints\resnet_model_best.pth
   ```

#### Issue: Streamlit won't start
**Solution:**
```powershell
# Reinstall Streamlit
pip uninstall streamlit
pip install streamlit

# Try different port
streamlit run app.py --server.port 8502

# Clear cache
streamlit cache clear
```

#### Issue: Image upload not working
**Solutions:**
1. Check image format (JPG, PNG supported)
2. Verify image is RGB (not RGBA or grayscale)
3. Check file size (very large images may fail)

### 🔴 TensorBoard Issues

#### Issue: TensorBoard won't start
**Solution:**
```powershell
# Reinstall
pip uninstall tensorboard
pip install tensorboard

# Try different port
tensorboard --logdir logs/tensorboard --port 6007

# Clear cache
Remove-Item -Recurse -Force $env:TEMP\.tensorboard-info\
```

#### Issue: No data in TensorBoard
**Solution:**
1. Verify logs exist:
   ```powershell
   dir logs\tensorboard
   ```

2. Train model to generate logs:
   ```powershell
   python main.py --mode train --epochs 5
   ```

3. Check correct log directory:
   ```powershell
   tensorboard --logdir logs/tensorboard
   ```

### 🔴 Import Errors

#### Issue: `ModuleNotFoundError`
**Solution:**
```powershell
# Verify installation
pip list | Select-String "torch|streamlit|matplotlib"

# Reinstall missing packages
pip install -r requirements.txt

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Issue: `ImportError: cannot import name`
**Solution:**
```powershell
# Update all packages
pip install --upgrade -r requirements.txt

# Check for version conflicts
pip check
```

### 🔴 File Permission Issues

#### Issue: Cannot write to checkpoint/logs/results
**Solution:**
```powershell
# Run as administrator or create directories manually
New-Item -ItemType Directory -Force -Path checkpoints
New-Item -ItemType Directory -Force -Path logs
New-Item -ItemType Directory -Force -Path results

# Or edit config.py to use different paths
```

### 🔴 Performance Issues

#### Issue: Inference is slow
**Solutions:**
1. Use GPU:
   ```python
   # In config.py
   DEVICE = torch.device('cuda')
   ```

2. Use smaller model (CNN instead of ResNet):
   ```powershell
   python main.py --model cnn
   ```

3. Reduce image size in config.py:
   ```python
   IMAGE_SIZE = 128  # Instead of 224
   ```

4. Enable half precision (if using GPU):
   ```python
   model.half()
   ```

### 🔴 Data Loading Issues

#### Issue: DataLoader is slow
**Solutions:**
1. Increase workers:
   ```powershell
   python main.py --num-workers 8
   ```

2. Enable pin_memory (already enabled for CUDA)

3. Reduce image preprocessing complexity

4. Use SSD instead of HDD for dataset storage

### 🔴 Reproducibility Issues

#### Issue: Results differ between runs
**Solution:**
```python
# Ensure in config.py:
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 🔧 Diagnostic Commands

### Check Environment
```powershell
# Python version
python --version

# PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Installed packages
pip list

# GPU information
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Check Dataset
```powershell
# Dataset statistics
python data.py

# Count images
python -c "from pathlib import Path; print(len(list(Path('Dataset/Training').rglob('*.jpg'))))"
```

### Check Models
```powershell
# Test model creation
python model.py

# Check checkpoint
python -c "import torch; print(torch.load('checkpoints/cnn_model_best.pth').keys())"
```

## 📞 Still Having Issues?

1. **Check error message carefully** - Most errors include helpful information

2. **Run examples.py** - Interactive examples help identify issues:
   ```powershell
   python examples.py
   ```

3. **Review PROJECT_SUMMARY.md** - Comprehensive overview of all components

4. **Check logs** - Review console output and log files

5. **Verify file structure** - Ensure all files are in correct locations

6. **Update packages** - Sometimes package updates fix issues:
   ```powershell
   pip install --upgrade -r requirements.txt
   ```

## 🐛 Debug Mode

Enable detailed debugging:

```python
# Add to your script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or run with Python debug mode
python -v main.py
```

## 💡 Best Practices to Avoid Issues

1. ✅ Always activate virtual environment
2. ✅ Use requirements.txt for consistent dependencies
3. ✅ Start with small experiments (few epochs)
4. ✅ Monitor system resources (RAM, GPU memory)
5. ✅ Save checkpoints frequently
6. ✅ Keep dataset organized and validated
7. ✅ Use version control (git)
8. ✅ Document any configuration changes

---

**Most issues can be resolved by checking the error message, verifying file paths, and ensuring dependencies are properly installed!** 🚀
