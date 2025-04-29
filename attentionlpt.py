import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# Placeholder for the PGD attack function. 
# This function should be implemented elsewhere and imported.
# It should take the model, input tensor x, and labels y, 
# and return the adversarial example x_adv.
import torch
import torch.nn.functional as F

def pgd_attack(model, x, y,
                  eps: float=1.0,      # L2 반경
                  alpha: float=0.1,    # 한 스텝 크기
                  iters: int=40,
                  random_start: bool=True,
                  clamp_min: float=0.0,
                  clamp_max: float=1.0):
    model.eval()
    x_adv = x.clone().detach()
    
    # 1) 랜덤 스타트: L2‐ball uniform
    if random_start:
        delta = torch.randn_like(x_adv)
        delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
        # unit vector
        delta = delta / (delta_norm.view(-1,1,1,1) + 1e-12)
        # 반경 eps 내로 scaling
        u = torch.rand(delta.size(0), device=x.device).view(-1,1,1,1)
        delta = delta * (u**(1/x_adv[0].numel())) * eps
        x_adv = torch.clamp(x_adv + delta, clamp_min, clamp_max).detach()
    
    for _ in range(iters):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # 2) grad 방향 정규화 (L2‐norm)
        grad_flat = grad.view(grad.size(0), -1)
        grad_norm = torch.norm(grad_flat, p=2, dim=1).view(-1,1,1,1)
        normalized_grad = grad / (grad_norm + 1e-12)
        
        # 3) 스텝 이동
        x_adv = x_adv + alpha * normalized_grad
        
        # 4) L2‐ball projection
        delta = x_adv - x
        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = torch.norm(delta_flat, p=2, dim=1).view(-1,1,1,1)
        factor = torch.clamp(eps / (delta_norm + 1e-12), max=1.0)
        delta = delta * factor
        
        x_adv = torch.clamp(x + delta, clamp_min, clamp_max).detach()
    
    return x_adv




class FeatureExtractor:
    """
    Helper class to register hooks and extract intermediate features.
    """
    def __init__(self, model: nn.Module, layer_names: List[str]):
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

    def get_features(self) -> Dict[str, torch.Tensor]:
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
    attack_kwargs: Optional[Dict] = None # dict | None -> Optional[Dict]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # tuple[...] -> Tuple[...]
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
    # Ensure model is in eval mode for PGD attack generation
    original_mode = model.training
    model.eval()
    x_adv = pgd_attack(model, x, y, **attack_kwargs)
    model.train(original_mode) # Restore original mode
    x_adv = x_adv.detach() # Ensure gradients don't flow back through attack generation

    # --- 디버깅 코드 추가 ---
    diff = (x_adv - x).abs().sum().item()
    print(f"[DEBUG] Difference between x_adv and x: {diff:.4f}")
    if diff == 0:
        print("[DEBUG] Warning: PGD attack did not modify the input!")
    # --- 디버깅 코드 끝 ---

    # 2) Forward pass for both clean and adversarial examples
    model.eval() # Ensure eval mode for consistent feature extraction if BN/Dropout are used
    
    # Clean forward pass
    logits_clean = model(x) # Run forward pass, populates hooks AND gets logits
    # --- 수정: 특징을 즉시 복사 ---
    feats_clean = {k: v.clone().detach() if v is not None else None 
                   for k, v in feature_extractor.get_features().items()}
    # --------------------------
    
    # Adversarial forward pass
    logits_adv = model(x_adv) # Run forward pass, populates hooks AND gets logits
    # --- 수정: 특징을 즉시 복사 ---
    feats_adv = {k: v.clone().detach() if v is not None else None 
                 for k, v in feature_extractor.get_features().items()}
    # --------------------------

    model.train(original_mode) # Restore training mode after forward passes if needed

    # Check if features were extracted correctly
    # Check against the layer names stored in the extractor instance
    if any(v is None for v in feats_clean.values()) or any(v is None for v in feats_adv.values()):
         problematic_layers_clean = [k for k, v in feats_clean.items() if v is None]
         problematic_layers_adv = [k for k, v in feats_adv.items() if v is None]
         raise RuntimeError(f"Feature extraction failed. Check layer names. \n" \
                            f"Requested layers: {feature_extractor.layer_names}. \n"
                            f"Layers returning None (clean): {problematic_layers_clean}. \n"
                            f"Layers returning None (adv): {problematic_layers_adv}. \n"
                            f"Available modules: {list(dict(model.named_modules()).keys())}")
         
    # Ensure layer names requested exist in both feature dictionaries
    layer_names_to_use = list(feature_extractor.layer_names)
    # Check based on the keys actually returned by get_features, which reflect successful hook registration
    clean_keys = set(feats_clean.keys())
    adv_keys = set(feats_adv.keys())
    missing_clean = [name for name in layer_names_to_use if name not in clean_keys]
    missing_adv = [name for name in layer_names_to_use if name not in adv_keys]

    if missing_clean or missing_adv:
        raise KeyError(f"One or more specified layer names not found in extracted features. \n"
                       f"Missing from clean features: {missing_clean}. \n"
                       f"Missing from adversarial features: {missing_adv}.")

    # 3) Cross-Entropy Loss (on adversarial examples)
    loss_ce = F.cross_entropy(logits_adv, y)

    # 4) Adversarial Logit Pairing (ALP) Loss
    loss_alp = F.mse_loss(logits_clean, logits_adv)
    # --- 추가 디버깅 --- 
    print(f"[DEBUG][compute_at_alp_loss] Calculated loss_alp: {loss_alp.item():.10f}") # Print immediately after calculation
    logit_diff = (logits_clean - logits_adv).abs().sum().item()
    print(f"[DEBUG] Difference between logits_clean and logits_adv: {logit_diff:.4f}")
    # -------------------

    # 5) Attention Map Pairing (AT) Loss
    loss_at = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    for name in layer_names_to_use:
        # Already checked above if the keys exist
        a_c = feats_clean[name]
        a_a = feats_adv[name]

        # This check might be redundant now due to the checks above, but kept for safety
        if a_c is None or a_a is None:
             print(f"Warning: Feature tensor for layer '{name}' is None. Skipping AT loss calculation for this layer.")
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
        current_at_loss = F.mse_loss(vec_c_normalized, vec_a_normalized)
        # --- 추가 디버깅 (layer1만 확인) --- 
        if name == layer_names_to_use[0]: # 첫번째 레이어만 확인
            feat_diff = (a_c - a_a).abs().sum().item()
            norm_vec_diff = (vec_c_normalized - vec_a_normalized).abs().sum().item()
            print(f"[DEBUG] Layer '{name}': Feat Diff: {feat_diff:.4f}, NormVec Diff: {norm_vec_diff:.4f}, Current AT Loss: {current_at_loss.item():.4f}")
        # --------------------------------
        loss_at += current_at_loss

    # 6) Combine losses
    total_loss = loss_ce + alpha * loss_alp + beta * loss_at
    
    # --- 추가 디버깅 ---
    print(f"[DEBUG][compute_at_alp_loss] Returning losses - CE: {loss_ce.item():.6f}, LP: {loss_alp.item():.10f}, AT: {loss_at.item():.6f}") # Print just before return
    # -------------------
    return total_loss, loss_ce, loss_alp, loss_at
