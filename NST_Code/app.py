"""Flask web application for the AdaIN neural style-transfer demo.

All project files are resolved relative to this file, so the app can be
started from Windows, macOS, or Linux without machine-specific paths.
"""

import os
import uuid
from pathlib import Path

import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, HiddenField, SubmitField
from wtforms.validators import InputRequired, NumberRange

from utils.models import Decoder, VGGEncoder
from utils.utils import adaptive_instance_normalization


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
EXAMPLES_DIR = BASE_DIR / "examples"
VGG_WEIGHTS = BASE_DIR / "vgg_normalised.pth"
DECODER_WEIGHTS = BASE_DIR / "experiment" / "final_exp" / "decoder_final.pth"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "local-development-key-change-me"),
    UPLOAD_FOLDER=str(UPLOAD_DIR),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB per request
)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    # These hidden fields allow a previously selected image to be retained
    # when the other image is re-selected.
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[InputRequired(), NumberRange(min=0.0, max=1.0)],
    )
    submit = SubmitField("Transfer Style")


def _load_state_dict(path: Path, device: torch.device):
    """Load a weights-only checkpoint and map it to the active device."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required model file was not found: {path}\n"
            "Make sure you extracted the complete project, including model weights."
        )
    return torch.load(path, map_location=device, weights_only=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file_storage) -> str:
    """Save an uploaded image under a unique, sanitized filename."""
    original_name = secure_filename(file_storage.filename or "")
    if not original_name or not allowed_file(original_name):
        raise ValueError("Please upload a PNG, JPG, or JPEG image.")

    extension = Path(original_name).suffix.lower()
    filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename
    file_storage.save(destination)

    # Verify the file is a readable image before trying model inference.
    try:
        with Image.open(destination) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise ValueError("The uploaded file is not a valid image.") from exc

    return filename


def style_transfer(content_image, style_image, alpha: float):
    content_transform = transforms.Compose(
        [transforms.Resize(512), transforms.ToTensor()]
    )
    style_transform = transforms.Compose(
        [transforms.Resize(512), transforms.ToTensor()]
    )

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_features = encoder(content_tensor, is_test=True)
        style_features = encoder(style_tensor, is_test=True)
        stylized_features = adaptive_instance_normalization(
            content_features, style_features
        )
        stylized_features = alpha * stylized_features + (1.0 - alpha) * content_features
        stylized_image = decoder(stylized_features)

    return stylized_image


def save_image_tensor(image_tensor, destination: Path) -> None:
    image_tensor = image_tensor.detach().cpu().squeeze(0).clamp(0, 1)
    transforms.ToPILImage()(image_tensor).save(destination)


# CPU is selected automatically when CUDA is unavailable (typical on laptops
# with integrated graphics). The pretrained weights are loaded portably.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VGGEncoder(VGG_WEIGHTS, map_location=device).to(device).eval()
decoder = Decoder().to(device)
decoder.load_state_dict(_load_state_dict(DECODER_WEIGHTS, device))
decoder.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        try:
            if form.content.data and form.content.data.filename:
                content_filename = _save_upload(form.content.data)
                form.content_path.data = content_filename
            else:
                content_filename = form.content_path.data or None

            if form.style.data and form.style.data.filename:
                style_filename = _save_upload(form.style.data)
                form.style_path.data = style_filename
            else:
                style_filename = form.style_path.data or None

            # Only accept previously stored names, not arbitrary paths supplied
            # in hidden form fields.
            if content_filename and (Path(content_filename).name != content_filename or "/" in content_filename or "\\" in content_filename):
                raise ValueError("Invalid content image selection. Please upload it again.")
            if style_filename and (Path(style_filename).name != style_filename or "/" in style_filename or "\\" in style_filename):
                raise ValueError("Invalid style image selection. Please upload it again.")

            if not content_filename or not style_filename:
                error = "Please upload both a content image and a style image."
            else:
                content_path = UPLOAD_DIR / content_filename
                style_path = UPLOAD_DIR / style_filename
                if not content_path.is_file() or not style_path.is_file():
                    raise ValueError("An uploaded image could not be found. Please upload both images again.")

                with Image.open(content_path) as image:
                    content_image = image.convert("RGB")
                with Image.open(style_path) as image:
                    style_image = image.convert("RGB")

                alpha = min(1.0, max(0.0, float(form.alpha.data)))
                stylized_tensor = style_transfer(content_image, style_image, alpha)
                result_filename = f"stylized_{uuid.uuid4().hex}.png"
                save_image_tensor(stylized_tensor, UPLOAD_DIR / result_filename)
                result_image = result_filename

        except (ValueError, OSError, RuntimeError) as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Unexpected error while generating the stylized image")
            error = "Something went wrong while generating the image. Check the terminal for details."

    elif form.is_submitted():
        error = "Please check the form and upload valid PNG, JPG, or JPEG images."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(EXAMPLES_DIR, filename)


@app.errorhandler(413)
def request_too_large(_error):
    return "The uploaded files are too large. Please keep the total request under 16 MB.", 413


if __name__ == "__main__":
    # Flask's development server works on Windows; Gunicorn is only used by
    # the separate Linux deployment configuration.
    app.run(host="127.0.0.1", port=5000, debug=False)
