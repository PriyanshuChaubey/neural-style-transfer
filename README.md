# Neural Style Transfer Using Adaptive Instance Normalization (AdaIN)

A Deep Learning-based Neural Style Transfer application that transforms ordinary images into artistic compositions by combining the content of one image with the visual style of another.

Built using **Python, PyTorch, VGG19, Adaptive Instance Normalization (AdaIN), and Flask**, this project provides an interactive web application for generating stylized images with adjustable style intensity.

---

## 1. Project Overview

Neural Style Transfer (NST) is a computer vision technique that combines the structural information of a content image with the artistic characteristics of a style image.

This project implements Neural Style Transfer using **Adaptive Instance Normalization (AdaIN)**, a technique that aligns the channel-wise mean and standard deviation of content features with those of style features.

A pretrained VGG encoder extracts visual features from the input images. AdaIN transforms the content features using the statistical properties of the style features, and a trained decoder reconstructs the final stylized image.

The project also includes a Flask-based web interface where users can upload images, control style intensity, generate artistic outputs, and download the results.

## 2. Key Features

- **Content and Style Image Upload:** Upload separate content and style-reference images.
- **Adaptive Instance Normalization:** Transfer artistic characteristics using feature-statistic alignment.
- **Pretrained VGG Encoder:** Extract deep visual representations from input images.
- **Trained Decoder Network:** Reconstruct stylized images from transformed feature representations.
- **Adjustable Style Intensity:** Control the balance between original content and transferred style using the Alpha parameter.
- **Interactive Web Application:** Generate stylized images through a Flask-based interface.
- **Image Preview and Download:** View uploaded images and download generated results.
- **Example Images:** Access sample images provided with the project.
- **Automatic Device Selection:** Use CUDA when available and fall back to CPU otherwise.
- **Cross-Platform Support:** Use project-relative paths for Windows, Linux, and macOS.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | Torchvision |
| Neural Network Architecture | VGG Encoder and Decoder |
| Style Transfer Technique | Adaptive Instance Normalization (AdaIN) |
| Backend Framework | Flask |
| Form Handling | Flask-WTF, WTForms |
| Image Processing | Pillow (PIL) |
| Numerical Computing | NumPy |
| Progress Monitoring | tqdm |
| Visualization | Matplotlib |
| Frontend | HTML5, CSS3, JavaScript |
| UI Framework | Bootstrap 5 |
| Icons | Font Awesome |
| Web Server | Gunicorn (Linux deployment) |
| Version Control | Git and GitHub |

## 4. How the Project Works

The application follows a feature-based image transformation pipeline.

### Step 1: Input Images

The user provides two images:

- **Content Image:** The original photograph whose structure and objects should be preserved.
- **Style Image:** The reference artwork whose visual characteristics should be transferred.

### Step 2: Feature Extraction Using VGG

A pretrained VGG-based encoder extracts feature representations from both images.

These feature representations capture important visual information such as structures, patterns, textures, and other image characteristics.

### Step 3: Adaptive Instance Normalization (AdaIN)

AdaIN aligns the statistical properties of content features with those of style features.

The transformation is expressed as:

\[
AdaIN(c,s)=\sigma(s)\left(\frac{c-\mu(c)}{\sigma(c)}\right)+\mu(s)
\]

Where:

- \(c\) represents content features.
- \(s\) represents style features.
- \(\mu\) represents the channel-wise mean.
- \(\sigma\) represents the channel-wise standard deviation.

### Step 4: Alpha-Based Feature Blending

The Alpha parameter controls the intensity of the style transfer.

\[
t=\alpha \cdot AdaIN(c,s)+(1-\alpha)\cdot c
\]

Where:

- Alpha = 0 retains the original content features.
- Alpha = 1 applies the fully style-normalized features.
- Intermediate values blend the original content with the transferred style.

### Step 5: Image Reconstruction

The decoder network reconstructs the final stylized image from the transformed feature representation.

### Step 6: Output Generation

The generated image is saved in the application's upload directory and displayed through the web interface. Users can preview and download the result.

## 5. System Architecture

The overall workflow of the application is:

```text
       Content Image       Style Image
             |                  |
             v                  v
        VGG Encoder        VGG Encoder
             |                  |
             v                  v
       Content Features    Style Features
             |                  |
             +--------+---------+
                      |
                      v
               AdaIN Transformation
                      |
                      v
              Alpha-Based Blending
                      |
                      v
                  Decoder
                      |
                      v
             Stylized Image Output
                      |
                      v
                Flask Web App
                      |
                      v
             Preview and Download
```

## 6. Project Structure

```text
neural-style-transfer/
│
├── NST_Code/
│   │
│   ├── app.py
│   ├── train.py
│   │
│   ├── utils/
│   │   ├── models.py
│   │   └── utils.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── static/
│   │   └── uploads/
│   │
│   ├── content_data/
│   ├── style_data/
│   ├── examples/
│   │
│   ├── experiment/
│   │   └── final_exp/
│   │       └── decoder_final.pth
│   │
│   └── vgg_normalised.pth
│
├── Demo_IO_Images/
├── code.ipynb
├── requirements.txt
├── requirements-deploy.txt
├── run_windows.bat
├── .gitignore
└── README.md
```

### Important Files

| File | Description |
|---|---|
| `app.py` | Flask backend, model loading, image uploads and inference |
| `train.py` | Decoder training pipeline |
| `models.py` | VGG encoder and decoder architecture |
| `utils.py` | Dataset loading, image transformations and AdaIN operations |
| `index.html` | Frontend interface |
| `vgg_normalised.pth` | Pretrained VGG encoder weights |
| `decoder_final.pth` | Trained decoder weights |
| `code.ipynb` | Notebook for VGG feature-map visualization |
| `requirements.txt` | Local development dependencies |
| `requirements-deploy.txt` | Additional Linux deployment dependency |

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or 3.11
- pip
- Git
- VS Code or another Python IDE

Python 3.11 is recommended for the pinned PyTorch and Torchvision versions.

### Step 1: Clone the Repository

```bash
git clone https://github.com/PriyanshuChaubey/neural-style-transfer.git
```

### Step 2: Navigate to the Project Directory

```bash
cd neural-style-transfer
```

### Step 3: Create a Virtual Environment

For Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

For Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

From the repository root:

```powershell
python .\NST_Code\app.py
```

Alternatively, on Windows, run:

```powershell
.\run_windows.bat
```

### Step 6: Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Upload a content image and a style image, adjust the Alpha parameter, and generate your stylized output.

## 8. Training the Decoder

The project includes a separate training pipeline implemented in `train.py`.

The training process uses content and style image datasets to optimize the decoder network.

### Training Workflow

1. Load content and style datasets.
2. Apply image resizing and preprocessing.
3. Extract feature representations using the pretrained VGG encoder.
4. Apply Adaptive Instance Normalization.
5. Generate reconstructed images using the decoder.
6. Calculate content and style losses.
7. Optimize decoder parameters using the Adam optimizer.
8. Save trained decoder and optimizer checkpoints.

### Run Training

From the repository root:

```powershell
python .\NST_Code\train.py --epochs 1
```

### Train Using Custom Datasets

```powershell
python .\NST_Code\train.py --content_dir "D:\datasets\content" --style_dir "D:\datasets\style" --epochs 10
```

### Additional Training Options

- `--no-crop`: Disable random cropping during preprocessing.
- `--resume`: Resume training from saved checkpoints.
- `--decoder_path`: Specify the decoder checkpoint.
- `--optimizer_path`: Specify the optimizer checkpoint.
- `--batch_size`: Configure the training batch size.
- `--lr`: Set the learning rate.
- `--epochs`: Set the number of training epochs.

**Note:** Training can be computationally expensive, particularly on CPU-only systems. The project already includes pretrained model weights, so retraining is not required to run the web application.

## 9. Pretrained Model Files

The application requires the following model files:

| Model File | Purpose |
|---|---|
| `vgg_normalised.pth` | Contains pretrained VGG encoder parameters |
| `decoder_final.pth` | Contains trained decoder parameters |

Keep both files in their original project-relative locations.

The application loads the model weights automatically when `app.py` starts.

## 10. Hardware and Compatibility

- **Operating Systems:** Windows 10/11, Linux and macOS.
- **Python:** 3.10 or 3.11 recommended.
- **CPU:** Supported for inference and training.
- **GPU:** CUDA acceleration is used automatically when a compatible NVIDIA GPU and CUDA-enabled PyTorch installation are available.

Integrated graphics generally do not support CUDA acceleration.

## 11. Deployment Notes

The project includes `requirements-deploy.txt`, which installs the additional Gunicorn dependency for Linux-based hosting.

For local Windows development, use `requirements.txt` and the Flask application command described above.

For a Linux deployment, the application can be served using Gunicorn with the application directory as its working directory.

## 12. Screenshots and Results

Example Markdown:

```markdown
[![Application Interface]](https://drive.google.com/file/d/1bMx5FuEVEYKTLlTf9O4xZinwZVXCJzfM/view?usp=sharing)

[![Style Transfer Result]](https://drive.google.com/file/d/150hzqR8mko1mIZ6rRAMtzljbrzfO4aJK/view?usp=sharing)
```
## 13. 🌐 Live Demo

Try the deployed application here:

**[Neural Style Transfer – Live Application](https://neural-style-transfer-1-3g1e.onrender.com)**

Upload a content image and a style image to generate a stylized output using Adaptive Instance Normalization (AdaIN).

## 14. Future Improvements

Potential areas for future development include:

- Support for additional image formats.
- Batch processing of multiple images.
- Additional artistic style presets.
- Improved inference performance.
- GPU acceleration and deployment optimization.
- Enhanced output comparison and visualization.
- Cloud-based model hosting.

## 15. Learning Outcomes

This project provided practical experience in:

- Deep Learning and Computer Vision.
- Convolutional Neural Networks.
- VGG-based feature extraction.
- Adaptive Instance Normalization.
- Content and style loss functions.
- PyTorch model training and inference.
- Model checkpoint management.
- Flask-based machine learning application development.
- Frontend and backend integration.
- Cross-platform Python development.

## 16. Acknowledgements

This project is based on the research paper:

**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**

Authors: Xun Huang and Serge Belongie  
Published at: IEEE International Conference on Computer Vision (ICCV), 2017.

The original research introduced Adaptive Instance Normalization for arbitrary image style transfer.

Please review the licensing and attribution requirements of the original code, pretrained model weights, and datasets before redistribution. Retain all required copyright and attribution notices.

## Author

**Priyanshu Chaubey**

B.Tech – Artificial Intelligence and Robotics

GitHub: [PriyanshuChaubey](https://github.com/PriyanshuChaubey)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
