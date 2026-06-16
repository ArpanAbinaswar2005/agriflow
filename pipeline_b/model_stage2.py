import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Tuple


class VisualEncoder(nn.Module):
    """Visual encoder using ResNet-50 backbone.

    Loads a pretrained ResNet-50 from torchvision, removes the final
    classification layer, and outputs a 2048-dimensional embedding.

    Input: (batch, 3, 224, 224) - fixed-rig camera image
    Output: (batch, 2048) - visual embedding
    """

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-50
        try:
            full_model = models.resnet50(pretrained=True)
        except TypeError:
            # Newer torchvision versions use weights enum
            full_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Keep layer4 for Grad-CAM
        self.layer4 = full_model.layer4

        # Remove the final classification layer (avgpool + fc)
        # Keep everything up to avgpool
        self.backbone = nn.Sequential(*list(full_model.children())[:-1])
        # Output from avgpool is (batch, 2048, 1, 1), we'll flatten it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224)

        Returns:
            embedding: (batch, 2048)
        """
        features = self.backbone(x)  # (batch, 2048, 1, 1)
        embedding = features.view(features.size(0), -1)  # (batch, 2048)
        return embedding


class ThermalEncoder(nn.Module):
    """Thermal encoder using 1D-CNN.

    Processes a sequence of temperature/humidity readings through 3 conv layers
    and outputs a 64-dimensional embedding.

    Input: (batch, seq_len, 2) - temperature and humidity readings over time
    Output: (batch, 64) - thermal embedding
    """

    def __init__(self, seq_len: int = 30):
        super().__init__()
        self.seq_len = seq_len

        # 1D convolutions along the sequence dimension
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 2)

        Returns:
            embedding: (batch, 64)
        """
        # Transpose to (batch, 2, seq_len) for 1D conv
        x = x.transpose(1, 2)

        x = self.relu(self.conv1(x))  # (batch, 16, seq_len)
        x = self.relu(self.conv2(x))  # (batch, 32, seq_len)
        x = self.relu(self.conv3(x))  # (batch, 64, seq_len)

        # Global average pooling
        x = self.pool(x)  # (batch, 64, 1)
        embedding = x.squeeze(-1)  # (batch, 64)
        return embedding


class PhysicalEncoder(nn.Module):
    """Physical encoder using MLP.

    Processes weight and density measurements through 2 hidden layers
    and outputs a 64-dimensional embedding.

    Input: (batch, 4) - [weight1, weight2, density1, density2] or similar physical measurements
    Output: (batch, 64) - physical embedding
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4) - physical measurements

        Returns:
            embedding: (batch, 64)
        """
        embedding = self.mlp(x)
        return embedding


class LateFusionGrader(nn.Module):
    """Late fusion grader combining visual, thermal, and physical encoders.

    Concatenates embeddings from all three encoders and passes through
    fully connected layers to predict 3 quality grades.

    Supports masked modality: if thermal or physical inputs are None,
    replaces with learned mask embedding of the same dimension.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.thermal_encoder = ThermalEncoder()
        self.physical_encoder = PhysicalEncoder()

        # Learned mask embeddings for missing modalities
        self.thermal_mask = nn.Parameter(torch.randn(64))
        self.physical_mask = nn.Parameter(torch.randn(64))

        # Concatenation: 2048 (visual) + 64 (thermal) + 64 (physical) = 2176
        fused_dim = 2048 + 64 + 64

        # Fusion MLP: 2 fully connected layers
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        image: torch.Tensor,
        thermal_data: Optional[torch.Tensor] = None,
        physical_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional masked modalities.

        Args:
            image: (batch, 3, 224, 224) - visual input (required)
            thermal_data: (batch, seq_len, 2) or None
            physical_data: (batch, 4) or None

        Returns:
            probs: (batch, 3) - class probabilities
        """
        batch_size = image.size(0)

        # Visual encoding (always present)
        visual_emb = self.visual_encoder(image)  # (batch, 2048)

        # Thermal encoding
        if thermal_data is not None:
            thermal_emb = self.thermal_encoder(thermal_data)  # (batch, 64)
        else:
            # Use learned mask embedding, replicate across batch
            thermal_emb = self.thermal_mask.unsqueeze(0).expand(batch_size, -1)

        # Physical encoding
        if physical_data is not None:
            physical_emb = self.physical_encoder(physical_data)  # (batch, 64)
        else:
            # Use learned mask embedding, replicate across batch
            physical_emb = self.physical_mask.unsqueeze(0).expand(batch_size, -1)

        # Concatenate embeddings
        fused = torch.cat([visual_emb, thermal_emb, physical_emb], dim=1)  # (batch, 2176)

        # Predict class logits
        logits = self.fusion_head(fused)  # (batch, 3)
        probs = self.softmax(logits)

        return probs


def get_verified_grade(
    model: LateFusionGrader,
    image: torch.Tensor,
    thermal_data: Optional[torch.Tensor] = None,
    physical_data: Optional[torch.Tensor] = None,
) -> Tuple[str, float]:
    """Get verified grade from multi-modal input.

    Args:
        model: LateFusionGrader instance
        image: (1, 3, 224, 224) or (3, 224, 224)
        thermal_data: (1, seq_len, 2) or (seq_len, 2) or None
        physical_data: (1, 4) or (4,) or None

    Returns:
        verified_grade (str): 'A', 'B', or 'C'
        confidence (float): probability for the grade
    """
    device = next(model.parameters()).device
    model.eval()

    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if thermal_data is not None and thermal_data.dim() == 2:
        thermal_data = thermal_data.unsqueeze(0)
    if physical_data is not None and physical_data.dim() == 1:
        physical_data = physical_data.unsqueeze(0)

    image = image.to(device)
    if thermal_data is not None:
        thermal_data = thermal_data.to(device)
    if physical_data is not None:
        physical_data = physical_data.to(device)

    with torch.no_grad():
        probs = model(image, thermal_data, physical_data)  # (1, 3)

    idx_to_grade = {0: "A", 1: "B", 2: "C"}
    best_idx = int(torch.argmax(probs[0]))
    verified_grade = idx_to_grade[best_idx]
    confidence = float(probs[0, best_idx].cpu().item())

    return verified_grade, confidence


def generate_gradcam(
    model: LateFusionGrader,
    image: torch.Tensor,
    target_class: int = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for the visual encoder.

    Computes gradients of the target class with respect to the final
    convolutional layer of ResNet-50 and produces a visualization.

    Args:
        model: LateFusionGrader instance
        image: (1, 3, 224, 224) or (3, 224, 224)
        target_class: class index (0, 1, or 2); if None, use predicted class

    Returns:
        heatmap: (224, 224) numpy array in range [0, 1]
    """
    device = next(model.parameters()).device
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    image.requires_grad = True

    # Get the final conv layer of ResNet-50 from the visual encoder
    visual_encoder = model.visual_encoder
    layer4 = visual_encoder.layer4

    # Forward pass to get activations
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # Register hooks on the last conv layer in layer4
    # layer4[-1] is the last Bottleneck block, conv3 is its last conv layer
    handle_fwd = layer4[-1].conv3.register_forward_hook(forward_hook)
    handle_bwd = layer4[-1].conv3.register_full_backward_hook(backward_hook)

    # Forward pass
    probs = model(image, None, None)

    if target_class is None:
        target_class = int(torch.argmax(probs[0]))

    # Backward pass for target class
    model.zero_grad()
    class_score = probs[0, target_class]
    class_score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    # Compute Grad-CAM
    # activations: (1, channels, height, width)
    # gradients: (1, channels, height, width)
    if activations is None or gradients is None:
        # Fallback: return uniform heatmap if hooks didn't work
        return np.ones((224, 224))

    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    heatmap = (activations * weights).sum(dim=1).squeeze(0)  # (H, W)
    heatmap = heatmap.cpu().numpy()

    # Normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    heatmap_max = heatmap.max()
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max

    # Resize to match input image size if needed
    if heatmap.shape != (224, 224):
        from scipy.ndimage import zoom

        scale = 224 / heatmap.shape[0]
        heatmap = zoom(heatmap, scale)

    # Final clipping to ensure strict [0, 1] range
    heatmap = np.clip(heatmap, 0, 1)

    return heatmap


if __name__ == "__main__":
    print("=== Testing Stage 2 Grader ===\n")

    # Initialize model
    model = LateFusionGrader()
    model.eval()

    batch_size = 1

    # Test 1: All modalities present
    print("Test 1: All three modalities present")
    image = torch.rand(batch_size, 3, 224, 224)
    thermal_data = torch.rand(batch_size, 30, 2)  # 30 timesteps, 2 sensors
    physical_data = torch.rand(batch_size, 4)  # 4 physical measurements

    grade, conf = get_verified_grade(model, image, thermal_data, physical_data)
    print(f"  Verified grade: {grade}, confidence: {conf:.3f}")

    # Test 2: Only visual (thermal and physical are None)
    print("\nTest 2: Only visual modality (thermal=None, physical=None)")
    grade_visual_only, conf_visual_only = get_verified_grade(model, image, None, None)
    print(f"  Verified grade: {grade_visual_only}, confidence: {conf_visual_only:.3f}")

    # Test 3: Grad-CAM visualization
    print("\nTest 3: Generating Grad-CAM heatmap")
    heatmap = generate_gradcam(model, image)
    print(f"  Heatmap shape: {heatmap.shape}, min: {heatmap.min():.3f}, max: {heatmap.max():.3f}")

    print("\n=== All tests completed successfully ===")
