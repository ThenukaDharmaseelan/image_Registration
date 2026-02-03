# 👁️ Retinal Image Registration Methods

Automated retinal vessel segmentation and image registration using deep learning.

## 📊 Dataset

This project uses the **FIRE (Fundus Image Registration) dataset** for retinal vessel analysis and image registration evaluation.
### FIRE Dataset Structure
```
FIRE/
├── Ground Truth/          # Ground truth homography annotations
├── Images/                # Fundus images
└── Masks/                 # Image masks
```
**Dataset Download:** https://projects.ics.forth.gr/cvrl/fire/

## 💻 Requirements

- **OS**: Linux or Mac (Windows users: install MinGW-w64)
- **GPU**: NVIDIA GPU with CUDA or Mac M2
- **Software**: Anaconda/Miniconda
- **Hardware**: 16GB+ GPU VRAM (tested on NVIDIA RTX 3090 with 24GB), 32GB+ System RAM recommended


## Method 1: 👁️EyeLiner

Automatic longitudinal image registration using fundus landmarks. EyeLiner is a deep learning pipeline for automatically aligning longitudinal fundus images using vessel and optic disk segmentation.

## 📊 Dataset

### FIRE Dataset Structure
```
data/retina_datasets/FIRE/
├── Disc/                  # Optic disc annotations
├── Ground Truth/          # Ground truth landmarks
├── images/                # Fundus images
├── Masks/                 # Image masks
├── vassal Seg/            # Vessel segmentations
└── fire_time_series.csv   # Dataset metadata
```

## 💻 Requirements

- **OS**: Linux or Mac (Windows users: install MinGW-w64)
- **GPU**: NVIDIA GPU with CUDA or Mac M2
- **Python**: 3.10.4 or higher
- **Software**: PyEnv and Poetry (or pip)
- **Hardware**: 16GB+ GPU VRAM (tested on NVIDIA RTX 3090 with 24GB), 32GB+ System RAM recommended

## 📋 Prerequisites

**Important:** EyeLiner requires vessel and optic disc segmentations as input. These segmentations must be generated using **AutoMorph** before running EyeLiner.

### Generate Segmentations with AutoMorph

1. **Setup AutoMorph** (see AutoMorph README for detailed installation)
```bash
## AutoMorph

Automated retinal vascular morphology quantification for vessel segmentation.

### Installation

1. **Create and activate environment**
```bash
conda create -n automorph python=3.11 -y
conda activate automorph
```

2. **Clone repository**
```bash
git clone https://github.com/ThenukaDharmaseelan/image_Registration.git
cd AutoMorph
```
3. **Install PyTorch** (check CUDA with `nvcc --version`)
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

4. **Install dependencies**
```bash
pip install --ignore-installed certifi
pip install -r requirement.txt
pip install efficientnet_pytorch==0.7.1 --no-deps
```

### Running the Code

**Step 1:** Place your FIRE dataset images in `./images` folder

**Step 2:** Generate resolution file
```bash
python generate_resolution.py
```

**Step 3:** Run the pipeline
```bash
sh run.sh
```

### Output
- Monitor GPU usage with `nvidia-smi`



5. **AutoMorph Output** - Use these for EyeLiner:
   - Vessel segmentations: `Results/M2/binary_vessel/`
   - Optic disc segmentations: `Results/M2/optic_disc_cup/`

Once you have the segmentations from AutoMorph, proceed with EyeLiner installation and execution.

## 🚀 Installation

### Option 1: Install via pip (Simplest)
```bash
pip install eyeliner
```

### Option 2: Install from GitHub with PyEnv and Poetry

1. **Install PyEnv**
```bash
curl https://pyenv.run | bash
```

2. **Setup environment**
```bash
pyenv install 3.10.4
pyenv virtualenv 3.10.4 eyeliner
pyenv activate eyeliner
```

3. **Clone repository**
```bash
https://github.com/ThenukaDharmaseelan/image_Registration.git
cd EyeLiner
```

4. **Install Poetry**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

5. **Install dependencies**
```bash
poetry install
```

## ▶️ Running the Code

### Quick Start (Using Bash Scripts)

**Step 1: Run Registration**
```bash
bash scripts/scripts_fire_experiments/fire_experiments_run.sh
```

**Step 2: Evaluate Results**
```bash
bash scripts/scripts_fire_experiments/fire_experiments_eval.sh
```

### Detailed Commands (Manual Execution)

**Step 1: Run Registration**
```bash
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/fire/tps1/splg-v \
--device cuda:0
```

**Parameters:**
- `-d`: Path to dataset CSV file
- `-f`, `-m`: Column names for fixed and moving images
- `-fv`, `-mv`: Column names for **vessel segmentation masks (from AutoMorph)**
- `-fd`, `-md`: Column names for **disk segmentation masks (from AutoMorph)**
- `-s`: Image size (256x256)
- `--input`: Input type (vessel)
- `--kp_method`: Keypoint detection method (splg - SuperPoint + LightGlue)
- `--reg_method`: Registration method (tps - Thin Plate Spline)
- `--lambda_tps`: Deformation control parameter (1)
- `--save`: Output directory
- `--device`: Computation device (cuda:0 or cpu)

**Step 2: Evaluate Results**
```bash
DATA=results/fire/tps1/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps1/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md None \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints manual_keypoints \
--save $SAVE \
--device cuda:0 \
--fire_eval
```

## 📤 Output

**After registration:**
```
results/fire/tps1/splg-v/
├── registration_params/              # TPS deformation fields (.pth files)
├── registration_keypoint_matches/    # Keypoint match visualizations
├── detected_keypoints/               # Detected keypoint coordinates
└── fire_time_series_results.csv      # Results CSV with registration paths
```

**After evaluation:**
```
results/fire/tps1/splg-v/
├── ckbd_images_after/                # Registration checkerboards (after)
├── ckbd_images_before/               # Registration checkerboards (before)
├── detected_keypoints/               # Keypoint visualizations
├── diff_map_images/                  # Difference/subtraction maps
├── flicker_images/                   # Flicker GIFs
├── registration_images/              # Registered moving images
├── registration_keypoint_matches/    # Keypoint match visualizations
├── registration_params/              # Registration parameters
├── seg_overlaps/                     # Segmentation overlaps
└── fire_time_series_results.csv      # Evaluation metrics
```

**Key Output Files:**
- 🔬 **Registered images** - in `registration_images/`
- 🎯 **Keypoint matches** - in `registration_keypoint_matches/`
- 📊 **Registration parameters** - in `registration_params/`
- 📈 **Evaluation metrics (MLE,AUC,Sucess Rate)** - in `fire_time_series_results.csv`

## 🔧 Common Issues

### Memory/RAM Errors

If you encounter out-of-memory errors:
- Reduce image size: change `-s 256` to `-s 128`
- Close other applications to free up GPU/RAM
- Monitor GPU usage with `nvidia-smi`
- Use CPU instead: `--device cpu`

### Missing Segmentation Files

If you get errors about missing vessel or disc masks:
- Ensure AutoMorph has completed successfully
- Verify segmentation paths in your CSV file match AutoMorph output locations
- Check that all images have corresponding segmentation files

  ## Method 2: 👁️ GeoFormer
  
## 📊 Dataset

### FIRE Dataset Structure
```
data/datasets/FIRE/
├── Ground Truth/          # Ground truth homography annotations
├── Images/                # Fundus images
└── Masks/                 # Image masks
```

  ## 🚀 Installation

1. **Create and activate environment**
```bash
conda create -n GeoFormer python==3.8 -y
conda activate GeoFormer
```
2. **Clone repository**
```bash
https://github.com/ThenukaDharmaseelan/image_Registration.git
cd GeoFormer
```
3. **Install dependencies**
```
pip install -r requirements.txt
```
## 📥 Download Pre-trained Model

Download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1giglxwMGlOb3qE-5AQ60R4495KxRGTqb) and place it in the `saved_ckpt/` folder:
```bash
mkdir -p saved_ckpt
# Place geoformer.ckpt in saved_ckpt/
```

## ▶️ Running the Code

### Homography Estimation on FIRE Dataset
```bash
python eval_FIRE.py
```

This will:
- Load the FIRE dataset images
- Perform homography estimation using GeoFormer
- Calculate evaluation metrics (MLE,AUC,Sucess Rate)


### Registered Images
```bash
python image_register.py
```
## 📤 Output

Results are saved in the `outputs/` directory:
```
outputs/  
└── fire_registered/       # Registered image outputs
    ├── registered_images/ # Warped/aligned images
    └── overlays/          # Registration quality visualizations

```
## Method 3: 👁️ RetinaRegNet

## 📊 Dataset
### FIRE Dataset Structure
```
FIRE/
├── Ground Truth/          # Ground truth homography annotations
├── Images/                # Fundus images
└── Masks/                 # Image masks
```

## 🚀 Installation

1. **Create and activate environment**
```bash
conda create -n retinaregnet python=3.10.12 -y
conda activate retinaregnet
```

2. **Clone repository**
```bash
git clone https://github.com/ThenukaDharmaseelan/image_Registration.git
cd RetinaRegNet
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```
**Required packages:**
- torch==2.0.1
- torchvision==0.15.2
- jax==0.4.23
- jaxlib==0.4.23
- accelerate==0.23.0
- diffusers==0.20.2
- transformers==4.34.0
- xformers==0.0.22
- numpy==1.23.5
- scipy==1.11.3
- opencv-python==4.10.0.84
- matplotlib==3.8.0

## ▶️ Running the Code

### Evaluation on FIRE Dataset
```bash
python retinaregnet_fire_evaluation_script.py
```

**Note:** 
- The script automatically processes all images and their corresponding ground truth landmarks
- Ensure sufficient GPU memory to avoid CUDA out-of-memory errors
- If running on limited resources, you may need to downscale image resolution

## 📤 Output

Results are saved in the `FIRE_Image_Registration_Results/` directory:
```
FIRE_Image_Registration_Results/
├── Final_Registration_Results/    # Final registered images and results
├── Stage1/                         # Intermediate Stage 1 results
├── Stage2/                         # Intermediate Stage 2 results
├── Landmark_Error_Plot.png         # Visualization of landmark errors
└── Success_Rate_Comparison.png     # Registration success rate comparison
```

**Key Output Files:**
- 🖼️ **Registered images** - Final aligned retinal images in `Final_Registration_Results/`
- 📊 **Landmark error plot** - Quantitative registration accuracy visualization
- 📈 **Success rate comparison** - Model performance comparison metrics
- 🔧 **Stage results** - Intermediate processing outputs for analysis

## Metrics Evaluation(NCC,NMI,Hausdorff Distance (95th percentile),Centerline Dice ,SSIM,Wasserstein Distance)

```bash
python evaluate.py
```












  






