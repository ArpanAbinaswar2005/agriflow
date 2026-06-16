import random
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image


class VideoFrameSampler:
    """Sample and preprocess frames from a video or list of images.

    Usage:
        sampler = VideoFrameSampler(source)
        frames_tensor = sampler.sample_frames()  # shape (10,3,224,224)

    Args:
        source: path to a video file (str) or a list of image file paths or PIL Images
        num_samples: number of frames to sample (default 10)
    """

    def __init__(self, source: Union[str, List[Union[str, Image.Image]]], num_samples: int = 10):
        self.source = source
        self.num_samples = num_samples
        # transform: resize to 224x224 and convert to tensor (C,H,W)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float),
        ])

    def _load_frames_from_list(self, frames_list: List[Union[str, Image.Image]]):
        imgs = []
        for f in frames_list:
            if isinstance(f, Image.Image):
                img = f.convert("RGB")
            else:
                img = Image.open(f).convert("RGB")
            imgs.append(img)
        return imgs

    def _load_frames_from_video(self, video_path: str):
        """Try to read video frames using torchvision; requires ffmpeg/av support."""
        try:
            from torchvision.io import read_video

            video, _, _ = read_video(video_path, pts_unit="sec")
            # video: (num_frames, H, W, C) uint8 tensor
            frames = [Image.fromarray(frame.numpy()) for frame in video]
            return frames
        except Exception:
            raise RuntimeError(
                "Reading video failed. Provide a list of image frames or ensure torchvision video support is installed."
            )

    def sample_frames(self) -> torch.Tensor:
        """Return a tensor of sampled frames with shape (num_samples, 3, 224, 224)."""
        # Load list of PIL Images
        if isinstance(self.source, (list, tuple)):
            imgs = self._load_frames_from_list(list(self.source))
        elif isinstance(self.source, str):
            imgs = self._load_frames_from_video(self.source)
        else:
            raise ValueError("Unsupported source type for VideoFrameSampler")

        if len(imgs) == 0:
            raise ValueError("No frames found in source")

        # Randomly sample frames (with replacement if fewer than needed)
        if len(imgs) >= self.num_samples:
            chosen = random.sample(imgs, self.num_samples)
        else:
            chosen = [random.choice(imgs) for _ in range(self.num_samples)]

        # Apply transforms and stack into tensor
        tensors = []
        to_tensor = T.ToTensor()
        for img in chosen:
            img = img.resize((224, 224))
            t = to_tensor(img)  # (3, H, W) float in [0,1]
            tensors.append(t)

        frames_tensor = torch.stack(tensors, dim=0)  # (num_samples, 3, 224, 224)
        return frames_tensor


class Stage1Grader(nn.Module):
    """Grader model using MobileNetV3-Large backbone.

    Loads a pretrained MobileNetV3-Large from torchvision and replaces the
    final classifier to predict 3 quality grades: A, B, C.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        # Load pretrained MobileNetV3-Large
        try:
            self.backbone = models.mobilenet_v3_large(pretrained=True)
        except TypeError:
            # For newer torchvision versions use weights enum
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        # Replace classifier head
        if hasattr(self.backbone, "classifier") and isinstance(self.backbone.classifier, nn.Sequential):
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            # Fallback: replace last layer if structure differs
            self.backbone.classifier = nn.Sequential(nn.Linear(1280, num_classes))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 3, 224, 224)

        Returns:
            probs: (batch, 3) class probabilities per frame
        """
        logits = self.backbone(x)
        probs = self.softmax(logits)
        return probs


def grade_batch(model: Stage1Grader, frames_tensor: torch.Tensor) -> Tuple[str, float, bool]:
    """Grade a batch of frames and return provisional grade, confidence, and low_confidence flag.

    Args:
        model: Stage1Grader instance
        frames_tensor: (num_frames, 3, 224, 224)

    Returns:
        provisional_grade (str): 'A', 'B', or 'C'
        confidence (float): probability for provisional grade (mean across frames)
        low_confidence (bool): True if entropy > 0.8
    """
    model.eval()
    device = next(model.parameters()).device
    frames = frames_tensor.to(device)

    with torch.no_grad():
        probs = model(frames)  # (num_frames, 3)

    # Average probabilities across frames
    mean_probs = probs.mean(dim=0).cpu().numpy()  # (3,)

    # Map index to grade
    idx_to_grade = {0: "A", 1: "B", 2: "C"}
    best_idx = int(np.argmax(mean_probs))
    provisional_grade = idx_to_grade[best_idx]
    confidence = float(mean_probs[best_idx])

    # Compute entropy for the mean distribution
    eps = 1e-9
    entropy = -float(np.sum(mean_probs * np.log(mean_probs + eps)))
    low_confidence = entropy > 0.8

    return provisional_grade, confidence, low_confidence


def get_price_premium(grade: str) -> float:
    """Return price premium percentage for a given grade.

    Grade A: +15%, B: 0%, C: -20%
    """
    grade = grade.upper()
    if grade == "A":
        return 0.15
    if grade == "B":
        return 0.0
    if grade == "C":
        return -0.20
    raise ValueError("Unknown grade: should be 'A', 'B' or 'C'")


class FarmerTrustScore:
    """Track farmer trust by comparing provisional vs verified grades.

    Stores history as a list of tuples (provisional, verified).
    get_score returns fraction of matches. is_reliable returns True if score>0.7.
    """

    def __init__(self):
        self.history: List[Tuple[str, str]] = []

    def update(self, provisional: str, verified: str):
        """Add a new (provisional, verified) pair to history."""
        self.history.append((provisional.upper(), verified.upper()))

    def get_score(self) -> float:
        """Return fraction of matches (provisional == verified)."""
        if not self.history:
            return 0.0
        matches = sum(1 for p, v in self.history if p == v)
        return matches / len(self.history)

    def is_reliable(self) -> bool:
        """Return True if trust score is above 0.7."""
        return self.get_score() > 0.7


if __name__ == "__main__":
    # Quick test using random fake frames
    # Create random frames tensor: (10,3,224,224)
    fake_frames = torch.rand(10, 3, 224, 224)

    # Initialize model and move to cpu
    grader = Stage1Grader()
    grader.eval()

    # Grade the fake frames
    grade, conf, low_conf = grade_batch(grader, fake_frames)
    print(f"Provisional grade: {grade}, confidence: {conf:.3f}, low_confidence: {low_conf}")

    # Price premium
    premium = get_price_premium(grade)
    print(f"Price premium for grade {grade}: {premium*100:.1f}%")

    # Farmer trust score demo
    fts = FarmerTrustScore()
    fts.update(grade, grade)  # pretend verified equals provisional
    print("Farmer trust score:", fts.get_score(), "Reliable?", fts.is_reliable())
