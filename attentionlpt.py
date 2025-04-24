import torch
import torch.nn as nn
import torch.nn.functional as F

# Placeholder for the PGD attack function. 
# This function should be implemented elsewhere and imported.
# It should take the model, input tensor x, and labels y, 
# and return the adversarial example x_adv.
def pgd_attack(model, x, y, **kwargs):
    """
    Placeholder for the PGD attack function.
    Replace this with your actual PGD implementation.
    """
    print("Warning: Using placeholder pgd_attack function. Replace with actual implementation.")
    # Example: simple random noise perturbation for placeholder
    delta = torch.randn_like(x) * 0.01 
    x_adv = torch.clamp(x + delta, 0, 1) # Assuming input is normalized [0, 1]
    return x_adv.detach()


class FeatureExtractor:
    """
    Helper class to register hooks and extract intermediate features.
    """
    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.model = model
        self.layer_names = layer_names
        self._features = {name: None for name in layer_names}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on specified layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._get_hook(name))
                self._hooks.append(hook)

    def _get_hook(self, name):
        """Returns a hook function that stores the output feature."""
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def get_features(self) -> dict[str, torch.Tensor]:
        """Returns the extracted features."""
        # Make sure features are cleared if the same extractor is used multiple times without re-registering
        # However, in typical training loops, a new forward pass populates them correctly.
        return self._features

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._features = {name: None for name in self.layer_names} # Clear features on removal

    def __del__(self):
        """Ensures hooks are removed when the object is deleted."""
        self.remove_hooks()

def compute_at_alp_loss(
    model: nn.Module, 
    feature_extractor: FeatureExtractor,
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float, 
    beta: float, 
    eps: float = 1e-8,
    attack_kwargs: dict | None = None # Optional kwargs for pgd_attack
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the combined AT+ALP loss.

    Args:
        model: The neural network model. Assumes model's forward pass returns logits.
        feature_extractor: An instance of FeatureExtractor attached to the model.
        x: Clean input batch tensor.
        y: Target labels tensor.
        alpha: Weight for the Adversarial Logit Pairing (ALP) loss.
        beta: Weight for the Attention Map Pairing (AT) loss.
        eps: Small epsilon value for numerical stability in normalization.
        attack_kwargs: Additional arguments to pass to the pgd_attack function.

    Returns:
        A tuple containing:
        - total_loss: The combined loss (scalar tensor).
        - loss_ce: Cross-entropy loss on adversarial examples.
        - loss_alp: Adversarial Logit Pairing loss.
        - loss_at: Attention Map Pairing loss.
    """
    if attack_kwargs is None:
        attack_kwargs = {}
        
    # 1) Generate adversarial example using PGD
    x_adv = pgd_attack(model, x, y, **attack_kwargs)
    x_adv = x_adv.detach() # Ensure gradients don't flow back through attack generation

    # 2) Forward pass for both clean and adversarial examples
    # Clean forward pass
    _ = model(x) # Run forward pass to populate hooks
    feats_clean = feature_extractor.get_features()
    # Re-run forward *without* hooks for clean logits if necessary, 
    # or modify model to return both logits and features
    # Assuming model(x) directly gives logits:
    logits_clean = model(x) 
    
    # Adversarial forward pass
    _ = model(x_adv) # Run forward pass to populate hooks
    feats_adv = feature_extractor.get_features()
    logits_adv = model(x_adv)

    # Check if features were extracted correctly
    if any(v is None for v in feats_clean.values()) or any(v is None for v in feats_adv.values()):
         raise RuntimeError(f"Feature extraction failed. Check layer names: {feature_extractor.layer_names}. Available modules: {list(dict(model.named_modules()).keys())}")
         
    # Ensure layer names requested exist in both feature dictionaries
    layer_names_to_use = list(feature_extractor.layer_names) # Use the names stored in the extractor
    if not all(name in feats_clean for name in layer_names_to_use) or \
       not all(name in feats_adv for name in layer_names_to_use):
        raise KeyError("One or more specified layer names not found in extracted features.")


    # 3) Cross-Entropy Loss (on adversarial examples)
    loss_ce = F.cross_entropy(logits_adv, y)

    # 4) Adversarial Logit Pairing (ALP) Loss
    # Ensure logits are compatible for mse_loss (e.g., same shape)
    loss_alp = F.mse_loss(logits_clean, logits_adv)

    # 5) Attention Map Pairing (AT) Loss
    loss_at = torch.tensor(0.0, device=x.device, dtype=x.dtype) # Initialize on correct device/dtype
    for name in layer_names_to_use:
        a_c = feats_clean[name]  # [B, C, H, W]
        a_a = feats_adv[name]

        if a_c is None or a_a is None:
             print(f"Warning: Feature tensor for layer '{name}' is None. Skipping.")
             continue

        # 5.1) Compute attention map (sum of absolute values across channels)
        att_c = a_c.abs().sum(dim=1)  # [B, H, W]
        att_a = a_a.abs().sum(dim=1)

        # 5.2) Flatten and normalize attention maps
        B = att_c.size(0)
        vec_c = att_c.view(B, -1)
        vec_a = att_a.view(B, -1)
        
        norm_c = vec_c.norm(p=2, dim=1, keepdim=True)
        norm_a = vec_a.norm(p=2, dim=1, keepdim=True)

        # Normalize, adding eps for stability
        vec_c_normalized = vec_c / (norm_c + eps)
        vec_a_normalized = vec_a / (norm_a + eps)

        # 5.3) Compute pairwise L2 distance (MSE Loss) between normalized vectors
        loss_at += F.mse_loss(vec_c_normalized, vec_a_normalized)

    # 6) Combine losses
    total_loss = loss_ce + alpha * loss_alp + beta * loss_at

    return total_loss, loss_ce, loss_alp, loss_at
