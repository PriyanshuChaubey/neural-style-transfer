"""Train the AdaIN decoder using paths relative to this project by default."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.models import Decoder, VGGEncoder
from utils.utils import (
    ImageFolderDataset,
    adaptive_instance_normalization,
    calc_mean_std,
    get_transform,
)


BASE_DIR = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the AdaIN style-transfer decoder")
    parser.add_argument(
        "--content_dir", type=Path, default=BASE_DIR / "content_data",
        help="Folder containing content images (defaults to NST_Code/content_data)",
    )
    parser.add_argument(
        "--style_dir", type=Path, default=BASE_DIR / "style_data",
        help="Folder containing style images (defaults to NST_Code/style_data)",
    )
    parser.add_argument(
        "--vgg", type=Path, default=BASE_DIR / "vgg_normalised.pth",
        help="Path to pretrained VGG weights",
    )
    parser.add_argument(
        "--experiment", type=str, default="experiment1",
        help="Name of the experiment output folder",
    )
    parser.add_argument("--final_size", type=int, default=256, help="Final crop/resize size")
    parser.add_argument("--content_size", type=int, default=512, help="Content image resize")
    parser.add_argument("--style_size", type=int, default=512, help="Style image resize")
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument("--crop", dest="crop", action="store_true", help="Use random crops (default)")
    crop_group.add_argument("--no-crop", dest="crop", action="store_false", help="Resize without random cropping")
    parser.set_defaults(crop=True)
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_decay", type=float, default=5e-5, help="Learning rate decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--content_weight", type=float, default=1.0, help="Content loss weight")
    parser.add_argument("--style_weight", type=float, default=5.0, help="Style loss weight")
    parser.add_argument("--log_interval", type=int, default=1, help="Log every N epochs")
    parser.add_argument("--save_interval", type=int, default=2, help="Save every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from supplied checkpoints")
    parser.add_argument("--decoder_path", type=Path, default=None, help="Decoder checkpoint for resume")
    parser.add_argument("--optimizer_path", type=Path, default=None, help="Optimizer checkpoint for resume")
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers (0 is the safest default on Windows)",
    )
    return parser.parse_args()


def _validate_args(args):
    for name in ("content_dir", "style_dir", "vgg"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"{name} path does not exist: {path}")
    if not args.content_dir.is_dir():
        raise NotADirectoryError(f"Content dataset must be a folder: {args.content_dir}")
    if not args.style_dir.is_dir():
        raise NotADirectoryError(f"Style dataset must be a folder: {args.style_dir}")
    if args.batch_size < 1 or args.epochs < 1 or args.num_workers < 0:
        raise ValueError("batch_size and epochs must be positive; num_workers cannot be negative")
    if args.log_interval < 1 or args.save_interval < 1:
        raise ValueError("log_interval and save_interval must be at least 1")
    if args.resume:
        if not args.decoder_path or not args.optimizer_path:
            raise ValueError("--resume requires both --decoder_path and --optimizer_path")
        if not args.decoder_path.is_file() or not args.optimizer_path.is_file():
            raise FileNotFoundError("One or both resume checkpoint files could not be found")


def main():
    args = parse_arguments()
    _validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("CPU mode selected. Training may be slow on a laptop CPU.")

    output_root = BASE_DIR / "experiment"
    save_dir = output_root / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "args.txt").open("w", encoding="utf-8") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key}: {value}\n")

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    if len(content_dataset) == 0 or len(style_dataset) == 0:
        raise ValueError("Both dataset folders must contain supported image files")

    # num_workers=0 avoids Windows multiprocessing/spawn surprises and is
    # appropriate for the small bundled sample datasets.
    pin_memory = device.type == "cuda"
    content_loader = DataLoader(
        content_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=pin_memory, drop_last=False, num_workers=args.num_workers,
    )
    style_loader = DataLoader(
        style_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=pin_memory, drop_last=False, num_workers=args.num_workers,
    )

    print(f"Content images: {len(content_dataset)} ({len(content_loader)} batches)")
    print(f"Style images:   {len(style_dataset)} ({len(style_loader)} batches)")

    encoder = VGGEncoder(args.vgg, map_location=device).to(device).eval()
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device, weights_only=True))
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device, weights_only=True))

    mse_loss = torch.nn.MSELoss()
    total_steps = min(len(content_loader), len(style_loader))
    print("Training...")

    for epoch in range(args.epochs):
        running_loss = running_content_loss = running_style_loss = 0.0
        progress = tqdm(zip(content_loader, style_loader), total=total_steps)

        for content_batch, style_batch in progress:
            content_batch = content_batch.to(device, non_blocking=pin_memory)
            style_batch = style_batch.to(device, non_blocking=pin_memory)

            content_features = encoder(content_batch)
            style_features = encoder(style_batch)
            target_features = adaptive_instance_normalization(
                content_features[-1], style_features[-1]
            )
            generated = decoder(target_features)
            generated_features = encoder(generated)

            content_loss = mse_loss(generated_features[-1], target_features) * args.content_weight
            style_loss = 0.0
            for generated_feature, style_feature in zip(generated_features, style_features):
                generated_mean, generated_std = calc_mean_std(generated_feature)
                style_mean, style_std = calc_mean_std(style_feature)
                style_loss = style_loss + mse_loss(generated_mean, style_mean) + mse_loss(generated_std, style_std)
            style_loss = style_loss * args.style_weight

            loss = content_loss + style_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_content_loss += content_loss.item()
            running_style_loss += style_loss.item()
            progress.set_description(
                f"Loss: {loss.item():.4f} | Content: {content_loss.item():.4f} | Style: {style_loss.item():.4f}"
            )

        scheduler.step()
        steps = max(1, total_steps)
        average_loss = running_loss / steps
        average_content_loss = running_content_loss / steps
        average_style_loss = running_style_loss / steps

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(
                f"Epoch {epoch + 1}: Loss={average_loss:.4f}, "
                f"Content={average_content_loss:.4f}, Style={average_style_loss:.4f}"
            )

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            torch.save(decoder.state_dict(), save_dir / f"decoder_{epoch + 1}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_{epoch + 1}.pth")
            with torch.no_grad():
                preview = torch.cat([content_batch, style_batch, generated], dim=0)
                save_image(preview, save_dir / f"output_{epoch + 1}.png", nrow=args.batch_size)

    print(f"Training complete. Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()
