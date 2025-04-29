# Sobra: Soybean disease classification and explanation using multi-modal data
This repository contains code for training, validating, and deploying the Streamlit web interface for Sobra. Follow the steps below to set up your environment, download the dataset, and run the application.

### Download the Dataset
Make sure you're on a Debian-based Linux system. Download the dataset in the directory containing the code.

```bash
# Install aria2 for fast, segmented downloading
sudo apt install aria2

# Download dataset using 16 parallel connections
aria2c -x 16 -s 16 -o data.rar "https://data.mendeley.com/public-files/datasets/w2r855hpx8/files/b027579b-9df8-4829-9ef2-27877a0990a7/file_downloaded"

# Extract the dataset
unrar x data.rar

# Ensure unrar is installed. If not, run:
sudo apt install unrar
```

### Install Python Dependencies
Create a virtual environment (optional but recommended), then install required packages:

```bash
# Create a new conda environment with Python 3.9 (or your desired version)
conda create -n soybean python=3.9 -y

# Activate the environment
conda activate soybean

# Install required packages
pip install -r requirements.txt
```

### Train the Model
To start training the model, run:
```bash
python train.py
```

### Validate the Model
To evaluate the model performance on the validation set:
```bash
python validation.py
```

### Run the Streamlit App
To launch the Streamlit web application:
```bash
streamlit run app.py
```

The app will open in your default browser.

## Notes
- Ensure all required files (dataset, model scripts, etc.) are in place before training or testing.
- Streamlit requires a compatible browser and a functioning Python environment.
- You can deactivate the conda environment anytime with `conda deactivate`.