# Quick Start Guide - Plant Disease Classification

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Verify Dataset Structure
Make sure your Dataset folder has this structure:
```
Dataset/
├── Training/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
├── Validation/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
└── Testing/
    ├── Early_Blight/
    ├── Healthy/
    └── Late_Blight/
```

### 3. Check Dataset Information
```powershell
python data.py
```

## 📊 Training Models

### Train Both Models (Recommended)
```powershell
python main.py --mode train --model all --epochs 50
```

### Train Only Custom CNN
```powershell
python main.py --mode train --model cnn --epochs 30
```

### Train Only ResNet50
```powershell
python main.py --mode train --model resnet --epochs 30
```

### Custom Training Parameters
```powershell
python main.py --mode train --model all --epochs 50 --batch-size 32 --learning-rate 0.001
```

## 🔍 Evaluating Models

### Evaluate Both Models
```powershell
python main.py --mode evaluate --model all --benchmark
```

### Evaluate Specific Model
```powershell
python main.py --mode evaluate --model resnet --benchmark
```

## 🎯 Complete Pipeline (Train + Evaluate)

### Run Everything
```powershell
python main.py --mode both --model all --epochs 50 --benchmark
```

## 📈 Monitoring Training

### Launch TensorBoard
```powershell
tensorboard --logdir logs/tensorboard
```
Then open: http://localhost:6006

## 🌐 Running the Web Application

### Launch Streamlit App
```powershell
streamlit run app.py
```
Then open: http://localhost:8501

## 🧪 Testing Individual Components

### Test Data Pipeline
```powershell
python data.py
```

### Test Model Architectures
```powershell
python model.py
```

### Test Training (2 epochs)
```powershell
python train.py
```

### Test Evaluation
```powershell
python evaluate.py
```

## 📁 Project Output

After training and evaluation, you'll find:

- **checkpoints/** - Saved model weights
  - `cnn_model_best.pth`
  - `resnet_model_best.pth`

- **logs/tensorboard/** - TensorBoard logs
  - Training/validation metrics
  - Loss and accuracy curves

- **results/** - Evaluation outputs
  - Confusion matrices
  - Classification reports
  - Sample predictions
  - Training history plots
  - `evaluation_results.json`

## ⚙️ Configuration

Edit `config.py` to customize:
- Batch size, learning rate, epochs
- Data augmentation parameters
- Model architecture settings
- File paths

## 💡 Common Use Cases

### Quick Test Run (5 epochs)
```powershell
python main.py --mode both --model cnn --epochs 5 --batch-size 16
```

### Production Training (GPU)
```powershell
python main.py --mode train --model all --epochs 100 --batch-size 64
```

### Evaluate Pretrained Models
```powershell
python main.py --mode evaluate --model all --benchmark --no-plot
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16`
- Use fewer workers: `--num-workers 2`

### Dataset Not Found
- Check Dataset folder structure
- Verify paths in `config.py`

### Model Checkpoint Not Found
- Train the model first before evaluation
- Check `checkpoints/` directory

## 📚 Next Steps

1. ✅ Train your models
2. ✅ Review TensorBoard visualizations
3. ✅ Check evaluation metrics in `results/`
4. ✅ Launch Streamlit app for predictions
5. ✅ Fine-tune hyperparameters if needed

## 🎓 Advanced Usage

### Resume Training from Checkpoint
Modify `train.py` to load checkpoint and continue training

### Custom Data Augmentation
Edit augmentation parameters in `config.py`

### Add New Disease Classes
1. Add images to Dataset folders
2. Update `CLASS_NAMES` in `config.py`
3. Retrain models

---

**Happy Training! 🌿🚀**
