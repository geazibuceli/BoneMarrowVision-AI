"""Experimental 128px class-conditioned GAN; synthetic samples are not reviewed labels."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image

from bone_marrow_vision.data.dataset import BoneMarrowDataset, sha256_file, verify_split_manifest
from bone_marrow_vision.training.checkpoints import atomic_torch_save, clone_state_dict
from bone_marrow_vision.training.recovery import loader_state, restore_loader
from bone_marrow_vision.training.reproducibility import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


class ConditionalGenerator(nn.Module):
    def __init__(self, classes: int, latent_dim: int = 128, width: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(classes, latent_dim)
        self.project = nn.Linear(latent_dim * 2, width * 16 * 4 * 4)
        layers = []
        channels = width * 16
        for output in (width * 8, width * 4, width * 2, width):
            layers.extend([
                nn.ConvTranspose2d(channels, output, 4, 2, 1, bias=False),
                nn.BatchNorm2d(output), nn.ReLU(True),
            ])
            channels = output
        layers.extend([nn.ConvTranspose2d(channels, 3, 4, 2, 1), nn.Tanh()])
        self.layers = nn.Sequential(*layers)
        self.width = width

    def forward(self, noise, labels):
        features = self.project(torch.cat((noise, self.embedding(labels)), dim=1))
        return self.layers(features.reshape(-1, self.width * 16, 4, 4))


class ConditionalDiscriminator(nn.Module):
    """Projection discriminator with spectral normalization and hinge loss."""

    def __init__(self, classes: int, width: int = 32):
        super().__init__()
        layers = []
        channels = 3
        for output in (width, width * 2, width * 4, width * 8, width * 16):
            layers.extend([
                spectral_norm(nn.Conv2d(channels, output, 4, 2, 1)),
                nn.LeakyReLU(0.2),
            ])
            channels = output
        self.features = nn.Sequential(*layers)
        self.score = spectral_norm(nn.Linear(channels, 1))
        self.embedding = spectral_norm(nn.Embedding(classes, channels))

    def forward(self, images, labels):
        features = self.features(images).mean(dim=(2, 3))
        return self.score(features).squeeze(1) + (features * self.embedding(labels)).sum(1)


def augmentation_plan(counts: dict[str, int], target: int) -> dict[str, int]:
    """At most one synthetic image per real training example, only below the target."""
    return {label: max(0, min(count, target - count)) for label, count in counts.items()}


def train_gan(data_dir: Path, output: Path, config: dict, *, resume: bool = False) -> Path:
    verify_split_manifest(data_dir)
    seed_everything(config["seed"])
    rows = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    if not rows or any(row.get("is_synthetic", False) for row in rows):
        raise ValueError("GAN training requires nonempty, real-only training data")
    names = sorted({row["label"] for row in rows})
    counts = Counter(row["label"] for row in rows)
    dataset = BoneMarrowDataset(rows, names)
    dataset.transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    samples_per_epoch = config["steps_per_epoch"] * config["batch_size"]
    sampler = WeightedRandomSampler(
        [1.0 / counts[row["label"]] for row in rows], samples_per_epoch,
        replacement=True, generator=make_generator(config["seed"]),
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], sampler=sampler,
        num_workers=config["num_workers"], worker_init_fn=seed_worker,
        generator=make_generator(config["seed"] + 1), pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    generator = ConditionalGenerator(len(names), config["latent_dim"], config["width"]).to(device)
    discriminator = ConditionalDiscriminator(len(names), config["width"]).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config["learning_rate"], betas=(0.0, 0.9))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config["learning_rate"], betas=(0.0, 0.9)
    )
    output.mkdir(parents=True, exist_ok=resume)
    fixed_labels = torch.arange(len(names), device=device).repeat_interleave(4)
    fixed_noise = torch.randn(len(fixed_labels), config["latent_dim"], device=device)
    history = []
    start_epoch = 1
    if resume and (output / "latest.pt").is_file():
        previous = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if previous.get("recovery_version") != 1:
            raise ValueError("This GAN checkpoint lacks complete recovery state")
        if (previous["config"] != config or previous["class_names"] != names
                or previous["training_split_sha256"] != sha256_file(data_dir / "train.json")
                or previous["split_manifest_sha256"] != sha256_file(data_dir / "manifest.json")):
            raise ValueError("GAN recovery inputs changed")
        generator.load_state_dict(previous["generator"])
        discriminator.load_state_dict(previous["discriminator"])
        g_optimizer.load_state_dict(previous["generator_optimizer"])
        d_optimizer.load_state_dict(previous["discriminator_optimizer"])
        fixed_noise = previous["fixed_noise"].to(device)
        fixed_labels = previous["fixed_labels"].to(device)
        history = previous["history"]
        start_epoch = previous["epoch"] + 1
        restore_loader(loader, previous["loader"])
        restore_rng_state(previous["rng"])
    for epoch in range(start_epoch, config["epochs"] + 1):
        generator.train()
        discriminator.train()
        g_total = d_total = 0.0
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            noise = torch.randn(len(real), config["latent_dim"], device=device)
            fake = generator(noise, labels)
            d_optimizer.zero_grad(set_to_none=True)
            d_loss = (
                nn.functional.relu(1 - discriminator(real, labels)).mean()
                + nn.functional.relu(1 + discriminator(fake.detach(), labels)).mean()
            )
            if not torch.isfinite(d_loss):
                raise RuntimeError("Non-finite GAN discriminator loss")
            d_loss.backward()
            d_optimizer.step()
            discriminator.requires_grad_(False)
            g_optimizer.zero_grad(set_to_none=True)
            g_loss = -discriminator(fake, labels).mean()
            if not torch.isfinite(g_loss):
                raise RuntimeError("Non-finite GAN generator loss")
            g_loss.backward()
            g_optimizer.step()
            discriminator.requires_grad_(True)
            g_total += g_loss.item()
            d_total += d_loss.item()
        entry = {"epoch": epoch, "generator_loss": g_total / len(loader),
                 "discriminator_loss": d_total / len(loader)}
        history.append(entry)
        print(json.dumps({"stage": "gan", **entry}), flush=True)
        generator.eval()
        with torch.inference_mode():
            save_image((generator(fixed_noise, fixed_labels) + 1) / 2,
                       output / f"preview_{epoch:03d}.png", nrow=4)
        payload = {
            "recovery_version": 1, "loader": loader_state(loader),
            "fixed_noise": fixed_noise.detach().cpu(), "fixed_labels": fixed_labels.detach().cpu(),
            "architecture": "conditional_gan_128", "generator": clone_state_dict(generator.state_dict()),
            "discriminator": clone_state_dict(discriminator.state_dict()),
            "generator_optimizer": g_optimizer.state_dict(),
            "discriminator_optimizer": d_optimizer.state_dict(),
            "rng": capture_rng_state(), "class_names": names, "config": config,
            "epoch": epoch, "history": history,
            "training_split_sha256": sha256_file(data_dir / "train.json"),
            "split_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        }
        atomic_torch_save(payload, output / "latest.pt")
        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            atomic_torch_save(payload, output / f"epoch_{epoch:03d}.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output / "latest.pt"


def generate_samples(checkpoint_path: Path, data_dir: Path, output: Path, seed: int, *, resume: bool = False) -> list[dict]:
    verify_split_manifest(data_dir)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if checkpoint["training_split_sha256"] != sha256_file(data_dir / "train.json"):
        raise ValueError("GAN checkpoint does not belong to this training split")
    rows = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    counts = Counter(row["label"] for row in rows)
    target = sorted(counts.values())[len(counts) // 2]
    plan = augmentation_plan(counts, target)
    config, names = checkpoint["config"], checkpoint["class_names"]
    seed_everything(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    generator = ConditionalGenerator(len(names), config["latent_dim"], config["width"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    output.mkdir(parents=True, exist_ok=resume)
    digest = sha256_file(checkpoint_path)
    samples = []
    real_hashes = {row["sha256"] for row in rows}
    seen = set(real_hashes)
    with torch.inference_mode():
        for class_id, label in enumerate(names):
            for batch in range(math.ceil(plan[label] / 32)):
                size = min(32, plan[label] - batch * 32)
                noise = torch.randn(size, config["latent_dim"], device=device)
                labels = torch.full((size,), class_id, dtype=torch.long, device=device)
                images = (generator(noise, labels) + 1) / 2
                for index, image in enumerate(images):
                    path = output / f"class_{class_id:02d}_{batch * 32 + index:05d}.png"
                    save_image(image, path)
                    image_hash = sha256_file(path)
                    if image_hash in seen:
                        raise ValueError("Exact duplicate synthetic image; inspect GAN collapse")
                    seen.add(image_hash)
                    samples.append({"path": str(path.resolve()), "label": label,
                                    "sha256": image_hash, "is_synthetic": True,
                                    "generator_sha256": digest, "generation_seed": seed,
                                    "generation_index": batch * 32 + index,
                                    "morphology_review": "unreviewed_experimental"})
    (output / "manifest.json").write_text(json.dumps({
        "generator_sha256": digest, "seed": seed, "target_real_count": target,
        "plan": plan, "samples": samples,
        "limitations": "Unreviewed morphology; byte-duplicate check is not a memorization audit.",
    }, indent=2), encoding="utf-8")
    return samples
