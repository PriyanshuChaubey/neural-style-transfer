# Neural Style Transfer (AdaIN)

A Flask web application that applies the style of one image to the content of another using Adaptive Instance Normalization (AdaIN), a pretrained VGG encoder, and a decoder network.

## Features

- Upload a content image and a style-reference image
- Adjust the style strength (`alpha`)
- Generate and download the stylized result
- Run inference on CPU or CUDA automatically
- Train the decoder using the bundled content and style image folders

## Project structure

```text
neural-style-transfer/
├── NST_Code/
│   ├── app.py
│   ├── train.py
│   ├── vgg_normalised.pth
│   ├── content_data/
│   ├── style_data/
│   ├── examples/
│   ├── experiment/final_exp/decoder_final.pth
│   ├── static/uploads/
│   ├── templates/index.html
│   └── utils/
├── Demo_IO_Images/
├── requirements.txt
├── requirements-deploy.txt
└── run_windows.bat
```

## Run on Windows 10/11

Python 3.10 or 3.11 is recommended. Python 3.11 is a good choice for the pinned PyTorch and Torchvision versions.

1. Extract or clone this repository.
2. Open PowerShell in the repository's root folder (the folder containing `requirements.txt`).
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead.

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Start the web application:

   ```powershell
   python .\NST_Code\app.py
   ```

   Or double-click `run_windows.bat` from File Explorer.

6. Open <http://127.0.0.1:5000> in your browser.

The app resolves the model files, examples, templates, and upload folder relative to `NST_Code/app.py`. It does not depend on a particular Windows username, drive letter, or current working directory.

## Run on Linux or macOS

Create a virtual environment, install `requirements.txt`, and run:

```bash
python NST_Code/app.py
```

Then open <http://127.0.0.1:5000>.

## Train the decoder

The default training paths are relative to `NST_Code`, so the bundled datasets are used automatically. Run from the repository root:

```powershell
python .\NST_Code\train.py --epochs 1
```

On a CPU-only laptop, training can be very slow. The bundled pretrained decoder is already available for the web demo, so you do not need to train the model just to run the application.

To use other datasets, pass their paths explicitly. For example:

```powershell
python .\NST_Code\train.py --content_dir "D:\datasets\content" --style_dir "D:\datasets\style" --epochs 10
```

To disable random cropping, add `--no-crop`. To resume training, pass `--resume` together with `--decoder_path` and `--optimizer_path`.

## Model files

The web app expects these pretrained files to exist:

- `NST_Code/vgg_normalised.pth`
- `NST_Code/experiment/final_exp/decoder_final.pth`

Keep these files in their existing relative locations when moving or cloning the project.

## Notes

- The application automatically selects CUDA when a compatible GPU is available; otherwise it uses CPU. Integrated graphics do not generally provide CUDA acceleration.
- `requirements.txt` is intended for local Windows/macOS/Linux development. `requirements-deploy.txt` additionally installs Gunicorn for Linux-based hosting; Gunicorn is not used by the Windows local run command.
- Uploaded images are stored under `NST_Code/static/uploads/` while the app is running. The Git ignore rules exclude generated uploads and Python virtual environments.
- Check the original code, pretrained-weight, and dataset licenses before redistributing the project publicly. Retain any required notices and attribution.

## Optional feature-visualization notebook

`code.ipynb` demonstrates VGG feature maps using the example images. Install the packages in `requirements.txt`, then open the notebook from the repository root in Jupyter or VS Code.
