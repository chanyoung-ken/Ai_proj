import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Optional

# --- Import from benchmark.py --- 
# Make sure benchmark.py is in the Python path or the same directory
try:
    from benchmark import (
        get_data_loader,
        get_base_resnet,
        fgsm_attack as benchmark_fgsm_attack, # Alias to avoid name clash
        pgd_attack as benchmark_pgd_attack,   # Alias to avoid name clash
        evaluate as benchmark_evaluate,       # Alias for clarity
        # Optionally import model definitions if needed elsewhere, though get_base_resnet is primary
        # ResNet_CBAM, 
        # ResNet_SE,
        DATASET_MEANS, # Import constants if needed
        DATASET_STDS
    )
    print("Successfully imported components from benchmark.py")
except ImportError as e:
    print(f"Error importing from benchmark.py: {e}")
    print("Please ensure benchmark.py is accessible.")
    # Exit or raise error if benchmark components are essential
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import from benchmark.py: {e}")
    exit()

# --- Import from our AT+ALP module ---
from attentionlpt import FeatureExtractor, compute_at_alp_loss, pgd_attack as alp_pgd_attack # Keep AT+ALP specific PGD

# --- Configuration (Adjust as needed) ---
CONFIG_FILE = '/home/work/AIprogramming/Ai_proj/best_hyperparams.json' # Benchmark config for LR/WD
DATASETS = ['CIFAR10'] # Focus dataset
# DATA_DIR is likely handled by get_data_loader from benchmark
BATCH_SIZE = 128
EPOCHS_AT_ALP = 50 # Epochs for AT+ALP training
USE_SUBSET = False # Set to True for faster testing (uses benchmark's SUBSET_SIZE if defined there, else needs definition)
# SUBSET_SIZE = 1000 # Define if benchmark doesn't expose it and USE_SUBSET is True
RESULTS_DIR = '/home/work/AIprogramming/Ai_proj/robustness_results' # Dir for benchmark models and saving results
ENSEMBLE_MODEL_DIR = '/home/work/AIprogramming/Ai_proj/ensemble_models' # Dir for AT+ALP trained models
BASE_MODEL_ARCH = 'resnet18'
OPTIMIZER_TYPE = 'SGD' # Default optimizer type

# AT+ALP Hyperparameters
ALPHA_ALP = 1.0 # Weight for Adversarial Logit Pairing loss
BETA_AT = 1.0 # Weight for Attention Map Pairing loss
AT_LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4'] # Example for ResNet18

# PGD Attack parameters specifically for AT+ALP training stage
ADV_EPS_TRAIN = 8/255
ADV_ALPHA_TRAIN = 2/255
ADV_ITERS_TRAIN = 7

# Evaluation attack parameters (using benchmark's style)
# These might be defined in benchmark.py, check there or redefine
ADV_EPS_EVAL = 0.25  # <-- 8/255 에서 0.25로 변경
ADV_ALPHA_EVAL = 2/255 # Alpha 값은 일단 유지 (필요시 조정 가능)
ADV_ITERS_EVAL = [7, 10, 20] # Test different PGD strengths for eval
FGSM_EPS_EVAL = [0.01, 0.03, 0.05, 0.07, 0.1] # Test different FGSM strengths

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Helper Functions (Reused or Adapted) ---

def load_model(model_path, arch, num_classes, device):
    """ Loads a model state dict from a path using benchmark's get_base_resnet. """
    # Use the imported function to get the model structure
    model = get_base_resnet(arch, num_classes)
    try:
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

# --- AT+ALP Training Function (Uses benchmark components where possible) ---

def train_at_alp(
    model, device, train_loader, test_loader, epochs, lr, weight_decay,
    optimizer_type, alpha_alp, beta_at, at_layer_names,
    attack_eps, attack_alpha, attack_iters, # Training attack params
    eval_attack_eps, eval_attack_alpha, eval_attack_iters, # Eval attack params
    model_save_path, model_name="AT_ALP_Model", dataset_name="Dataset"
):
    """ Trains a model using AT+ALP loss, evaluates with benchmark's evaluate and PGD attack. """
    model.to(device)

    # Setup optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Default to SGD if type is unknown, or raise error
        print(f"Warning: Unsupported optimizer type '{optimizer_type}'. Defaulting to SGD.")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup Feature Extractor for AT loss
    feature_extractor = FeatureExtractor(model, at_layer_names)

    best_robust_acc = 0.0
    best_weights = None
    start_time = time.time()

    print(f"\n--- Starting AT+ALP Training for {model_name} on {dataset_name} ---")
    # ... [logging prints remain the same] ...
    print(f"Hyperparameters: LR={lr}, WD={weight_decay}, Optimizer={optimizer_type}")
    print(f"AT+ALP Params: Alpha(ALP)={alpha_alp}, Beta(AT)={beta_at}, Layers={at_layer_names}")
    print(f"Training Attack: PGD(eps={attack_eps:.4f}, alpha={attack_alpha:.4f}, iters={attack_iters}) using alp_pgd_attack")
    print(f"Evaluation Attack: PGD(eps={eval_attack_eps:.4f}, alpha={eval_attack_alpha:.4f}, iters={eval_attack_iters}) using benchmark_pgd_attack")
    print(f"Total Epochs: {epochs}")
    
    criterion_eval = nn.CrossEntropyLoss() # For benchmark_evaluate if it needs it for attacks

    for epoch in range(epochs):
        model.train()
        total_loss_accum = 0.0
        ce_loss_accum = 0.0
        alp_loss_accum = 0.0
        at_loss_accum = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            # Compute AT+ALP loss using alp_pgd_attack internally
            attack_kwargs = {'eps': attack_eps, 'alpha': attack_alpha, 'iters': attack_iters}
            total_loss, loss_ce, loss_alp, loss_at = compute_at_alp_loss(
                model, feature_extractor, images, labels,
                alpha=alpha_alp, beta=beta_at,
                attack_kwargs=attack_kwargs # Uses attentionlpt.pgd_attack
            )

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            # --- 추가: Gradient Clipping --- 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm 값은 조절 가능
            optimizer.step()

            total_loss_accum += total_loss.item()
            # ... [accumulate other losses] ...
            ce_loss_accum += loss_ce.item()
            alp_loss_accum += loss_alp.item()
            at_loss_accum += loss_at.item()

            if i % 50 == 0: # Log progress
                 progress_bar.set_postfix({
                     'Loss': f"{total_loss.item():.3f}", # Use f-string
                     'CE': f"{loss_ce.item():.3f}",
                     'ALP': f"{loss_alp.item():.3f}",
                     'AT': f"{loss_at.item():.3f}"
                 })

        scheduler.step()

        avg_total_loss = total_loss_accum / len(train_loader)
        # ... [calculate avg other losses] ...
        avg_ce_loss = ce_loss_accum / len(train_loader)
        avg_alp_loss = alp_loss_accum / len(train_loader)
        avg_at_loss = at_loss_accum / len(train_loader)

        # Evaluate robust accuracy using benchmark's evaluate and benchmark's PGD
        # Ensure benchmark_evaluate signature matches: evaluate(model, device, loader, attack=None, attack_name=None, **attack_params)
        robust_acc = benchmark_evaluate(model, device, test_loader,
                                        attack=benchmark_pgd_attack, # Use benchmark's PGD
                                        attack_name=f'PGD_Eval(i={eval_attack_iters})', # Naming convention
                                        epsilon=eval_attack_eps,
                                        alpha=eval_attack_alpha,
                                        iters=eval_attack_iters
                                       )

        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Avg Loss: {avg_total_loss:.4f} (CE: {avg_ce_loss:.4f}, ALP: {avg_alp_loss:.4f}, AT: {avg_at_loss:.4f})")
        print(f"  Robust Accuracy (PGD-{eval_attack_iters} @ eps={eval_attack_eps:.3f}): {robust_acc:.4f}") # Use eval params in log

        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_save_path)
            print(f"  ** New best robust accuracy! Model saved to {model_save_path} **")

    total_time = time.time() - start_time
    print(f"--- AT+ALP Training Completed for {model_name} ---")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Best Robust Accuracy Achieved: {best_robust_acc:.4f}")

    # Load best weights back
    if best_weights:
        model.load_state_dict(best_weights)

    # Cleanup hooks
    feature_extractor.remove_hooks()

    return model, best_robust_acc, total_time

# --- Ensemble Prediction Function (Fix type hints) ---
def ensemble_predict(models: List[nn.Module], device: torch.device, loader: DataLoader, return_logits: bool = True):
    """
    Performs inference using an ensemble of models by averaging their logits or probabilities.
    (Keep this function as is, it operates on loaded model objects)
    """
    if not models:
        raise ValueError("Model list cannot be empty for ensemble prediction.")

    all_outputs = []
    all_labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Ensemble Predicting", leave=False):
            images = images.to(device)
            # labels = labels.to(device) # Labels needed only if return_logits=False
            
            batch_logits = []
            for model in models:
                # Ensure models are in eval mode
                model.eval()
                outputs = model(images)
                batch_logits.append(outputs)
                
            # Stack logits and average
            stacked_logits = torch.stack(batch_logits, dim=0)
            avg_logits = torch.mean(stacked_logits, dim=0)
            
            all_outputs.append(avg_logits.cpu()) # Store avg logits
            if not return_logits:
                all_labels_list.append(labels.cpu()) # Store labels if preds are needed

    if return_logits:
        return all_outputs # List of tensors [B, NumClasses]
    else:
        # Concatenate results
        all_logits_cat = torch.cat(all_outputs, dim=0)
        all_labels_cat = torch.cat(all_labels_list, dim=0)
        
        # Get final predictions
        final_preds = all_logits_cat.argmax(dim=1)
        return final_preds, all_labels_cat


# --- Ensemble Robustness Evaluation (Fix type hints) ---
def evaluate_robustness_ensemble(
    model_paths: List[str],
    base_arch: str,
    num_classes: int,
    device: torch.device,
    test_loader: DataLoader,
    dataset_name: str,
    results_save_dir: str
):
    """
    Evaluates the clean accuracy and adversarial robustness of an ensemble model
    and its individual components using benchmark's evaluate and attacks.
    """
    print(f"\n--- Evaluating Ensemble Robustness on {dataset_name} ---")
    print(f"Models in ensemble: {len(model_paths)}")

    # Load models using our helper that uses benchmark.get_base_resnet
    models = []
    model_names = []
    for path in model_paths:
        model = load_model(path, base_arch, num_classes, device)
        if model:
            models.append(model)
            model_names.append(os.path.basename(path).replace('.pth', ''))
        else:
            print(f"Skipping model due to loading error: {path}")

    if not models:
        print("No models were loaded successfully. Aborting evaluation.")
        return

    # --- Evaluation Setup ---
    criterion = nn.CrossEntropyLoss() # Required by benchmark attack functions
    results = {} # Store results

    # --- Evaluate Ensemble ---
    print("\nEvaluating Ensemble Model...")
    ensemble_name = "Ensemble"
    results[ensemble_name] = {}

    # Clean Accuracy (Ensemble) - Use ensemble_predict
    ensemble_preds, ensemble_labels = ensemble_predict(models, device, test_loader, return_logits=False)
    clean_acc_ensemble = (ensemble_preds == ensemble_labels).sum().item() / len(ensemble_labels)
    results[ensemble_name]['CleanAccuracy'] = clean_acc_ensemble
    print(f"{ensemble_name} Clean Accuracy: {clean_acc_ensemble:.4f}")

    # Adversarial Robustness (Ensemble) - FGSM using benchmark_fgsm_attack
    print("Evaluating Ensemble FGSM Robustness (using benchmark_fgsm_attack)...")
    results[ensemble_name]['FGSM'] = {}
    for eps in FGSM_EPS_EVAL:
        correct, total = 0, 0
        for images, labels in tqdm(test_loader, desc=f"Ensemble FGSM (eps={eps})", leave=False):
            images, labels = images.to(device), labels.to(device)
            # Generate attack using the first model and benchmark's FGSM
            with torch.enable_grad():
                adv_images = benchmark_fgsm_attack(models[0], criterion, images, labels, epsilon=eps, device=device)

            # Evaluate ensemble on these adversarial images
            batch_logits_adv = []
            with torch.no_grad():
                for model in models:
                    model.eval()
                    outputs = model(adv_images)
                    batch_logits_adv.append(outputs)
                stacked_logits_adv = torch.stack(batch_logits_adv, dim=0)
                avg_logits_adv = torch.mean(stacked_logits_adv, dim=0)
                preds_adv = avg_logits_adv.argmax(dim=1)
                correct += (preds_adv == labels).sum().item()
                total += labels.size(0)

        fgsm_acc = correct / total
        results[ensemble_name]['FGSM'][f"eps={eps:.2f}"] = fgsm_acc
        print(f"  FGSM (eps={eps:.2f}): Accuracy = {fgsm_acc:.4f}")

    # Adversarial Robustness (Ensemble) - PGD using benchmark_pgd_attack
    print("Evaluating Ensemble PGD Robustness (using benchmark_pgd_attack)...")
    results[ensemble_name]['PGD'] = {}
    for iters in ADV_ITERS_EVAL:
        correct, total = 0, 0
        eps = ADV_EPS_EVAL
        alpha = ADV_ALPHA_EVAL
        for images, labels in tqdm(test_loader, desc=f"Ensemble PGD (iters={iters})", leave=False):
            images, labels = images.to(device), labels.to(device)
            # Generate attack using the first model and benchmark's PGD
            with torch.enable_grad():
                adv_images = benchmark_pgd_attack(models[0], criterion, images, labels, epsilon=eps, alpha=alpha, iters=iters, device=device)

            # Evaluate ensemble
            batch_logits_adv = []
            with torch.no_grad():
                for model in models:
                    model.eval()
                    outputs = model(adv_images)
                    batch_logits_adv.append(outputs)
                stacked_logits_adv = torch.stack(batch_logits_adv, dim=0)
                avg_logits_adv = torch.mean(stacked_logits_adv, dim=0)
                preds_adv = avg_logits_adv.argmax(dim=1)
                correct += (preds_adv == labels).sum().item()
                total += labels.size(0)

        pgd_acc = correct / total
        results[ensemble_name]['PGD'][f"iters={iters}"] = pgd_acc
        print(f"  PGD (iters={iters}, eps={eps:.3f}): Accuracy = {pgd_acc:.4f}")

    # --- Evaluate Individual Models using benchmark_evaluate ---
    print("\nEvaluating Individual Models (using benchmark_evaluate)...")
    for i, model in enumerate(models):
        name = model_names[i]
        print(f"Evaluating {name}...")
        results[name] = {}

        # Clean Accuracy using benchmark_evaluate
        clean_acc_ind = benchmark_evaluate(model, device, test_loader)
        results[name]['CleanAccuracy'] = clean_acc_ind
        print(f"  {name} Clean Accuracy: {clean_acc_ind:.4f}")

        # FGSM Robustness using benchmark_evaluate and benchmark_fgsm_attack
        results[name]['FGSM'] = {}
        for eps_f in FGSM_EPS_EVAL:
            fgsm_acc_ind = benchmark_evaluate(model, device, test_loader,
                                              attack=benchmark_fgsm_attack,
                                              attack_name=f'FGSM (eps={eps_f:.2f})',
                                              epsilon=eps_f)
            results[name]['FGSM'][f"eps={eps_f:.2f}"] = fgsm_acc_ind

        # PGD Robustness using benchmark_evaluate and benchmark_pgd_attack
        results[name]['PGD'] = {}
        eps_p = ADV_EPS_EVAL
        alpha_p = ADV_ALPHA_EVAL
        for iters_p in ADV_ITERS_EVAL:
            pgd_acc_ind = benchmark_evaluate(model, device, test_loader,
                                             attack=benchmark_pgd_attack,
                                             attack_name=f'PGD (iters={iters_p})',
                                             epsilon=eps_p,
                                             alpha=alpha_p,
                                             iters=iters_p)
            results[name]['PGD'][f"iters={iters_p}"] = pgd_acc_ind
        print(f"  {name} evaluation complete.")

    # --- Save Results ---
    # (Keep the saving logic as is, it formats the results dictionary)
    print("\nSaving evaluation results...")
    df_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name, 'Type': 'Ensemble' if model_name == ensemble_name else 'Individual'}
        row['CleanAccuracy'] = metrics.get('CleanAccuracy', float('nan'))
        fgsm_metrics = metrics.get('FGSM', {})
        for k, v in fgsm_metrics.items():
            row[f'FGSM_{k}'] = v
        pgd_metrics = metrics.get('PGD', {})
        for k, v in pgd_metrics.items():
            row[f'PGD_{k}_eps{ADV_EPS_EVAL:.3f}'] = v # Make PGD column name clearer
        df_data.append(row)
    df_results = pd.DataFrame(df_data)
    results_filename = os.path.join(results_save_dir, f"{dataset_name}_all_ensemble_robustness_summary_compat.csv") # New name
    df_results.to_csv(results_filename, index=False)
    print(f"Evaluation results saved to {results_filename}")

    print("--- Ensemble Evaluation Complete ---")
    return df_results

# --- Main Execution Logic (Uses benchmark components) ---

if __name__ == '__main__':
    # Basic Setup
    dataset_name = DATASETS[0]

    # Load data loaders using benchmark function
    print(f"Loading dataset: {dataset_name} using benchmark.get_data_loader")
    # Pass necessary args expected by benchmark's get_data_loader (e.g., DATA_DIR, BATCH_SIZE)
    # Assuming benchmark defines DATA_DIR, BATCH_SIZE constants or handles them internally
    # If not, pass them: get_data_loader(dataset_name, BATCH_SIZE, train=True, data_dir=DATA_DIR, use_subset=USE_SUBSET)
    try:
        train_loader, num_classes = get_data_loader(dataset_name, BATCH_SIZE, train=True)
        test_loader, _ = get_data_loader(dataset_name, BATCH_SIZE, train=False)
        print(f"Number of classes: {num_classes}")
    except NameError as e:
        print(f"Error calling get_data_loader: {e}. Check if required constants (BATCH_SIZE, DATA_DIR) are defined or imported.")
        exit()
    except Exception as e:
        print(f"Error during data loading: {e}")
        exit()
        
    # --- Option 1: Train a new AT+ALP model ---
    at_alp_model_filename = f"{dataset_name}_at_alp_{BASE_MODEL_ARCH}_pgd{ADV_ITERS_TRAIN}.pth"
    at_alp_model_path = os.path.join(ENSEMBLE_MODEL_DIR, at_alp_model_filename)

    # Instantiate model using benchmark function
    print(f"Initializing model {BASE_MODEL_ARCH} using benchmark.get_base_resnet...")
    at_alp_model = get_base_resnet(BASE_MODEL_ARCH, num_classes)

    # Load hyperparameters (LR, WD, Optimizer type)
    try:
        with open(CONFIG_FILE) as f:
            best_params = json.load(f)
        cfg = best_params.get(dataset_name, {})
        # Use defaults matching benchmark script if possible
        lr = cfg.get('lr', 0.01) # Use benchmark's default or config
        wd = cfg.get('wd', 5e-4)
        opt_type = cfg.get('opt', OPTIMIZER_TYPE) # Get optimizer from config or use default
        print(f"Using training params for AT+ALP: LR={lr}, WD={wd}, Optimizer={opt_type}")
    except FileNotFoundError:
        print(f"Warning: Config file {CONFIG_FILE} not found. Using default LR=0.01, WD=5e-4, Optimizer={OPTIMIZER_TYPE}.")
        lr, wd, opt_type = 0.01, 5e-4, OPTIMIZER_TYPE
    except Exception as e:
         print(f"Error reading config file {CONFIG_FILE}: {e}. Using defaults.")
         lr, wd, opt_type = 0.01, 5e-4, OPTIMIZER_TYPE

    # Run AT+ALP training
    TRAIN_NEW_MODEL = False # Set to False to skip training if model exists
    if TRAIN_NEW_MODEL or not os.path.exists(at_alp_model_path):
        print(f"Starting AT+ALP training for {at_alp_model_filename}...")
        # Pass evaluation attack parameters for the eval step within training
        train_at_alp(
            at_alp_model, device, train_loader, test_loader, EPOCHS_AT_ALP, lr, wd,
            opt_type, ALPHA_ALP, BETA_AT, AT_LAYER_NAMES,
            ADV_EPS_TRAIN, ADV_ALPHA_TRAIN, ADV_ITERS_TRAIN, # Training PGD params
            ADV_EPS_EVAL, ADV_ALPHA_EVAL, ADV_ITERS_EVAL[0], # Pass benchmark PGD params for evaluation during training loop
            at_alp_model_path, model_name=at_alp_model_filename.replace('.pth',''), dataset_name=dataset_name
        )
    else:
        print(f"Skipping training for {at_alp_model_filename}, file already exists.")

    # --- Option 2: Define list of models for ensemble ---
    model_paths_for_ensemble = [
        # Paths from benchmark results (ensure filenames match benchmark's saving format)
        # os.path.join(RESULTS_DIR, f"{dataset_name}_baseline_resnet.pth"),
        os.path.join(RESULTS_DIR, f"{dataset_name}_resnet_cbam.pth"), # Uncomment to include
        os.path.join(RESULTS_DIR, f"{dataset_name}_resnet_se.pth"),   # Uncomment to include
        os.path.join(RESULTS_DIR, f"{dataset_name}_adv_resnet.pth"),
        at_alp_model_path # Include the AT+ALP trained model
    ]

    # Filter out paths that don't exist
    existing_model_paths = [p for p in model_paths_for_ensemble if os.path.exists(p)]
    if len(existing_model_paths) < len(model_paths_for_ensemble):
        print("Warning: Some specified model paths do not exist and will be excluded.")
        print("Final models included in ensemble evaluation:")
        for p in existing_model_paths:
             print(f" - {os.path.basename(p)}")

    if not existing_model_paths:
        print("Error: No valid model paths found for the ensemble. Exiting.")
    elif len(existing_model_paths) < 2:
         print("Warning: Only one model found. Ensemble evaluation will run, but results might not be meaningful.")
         # Run evaluation even for a single model for consistency
         evaluate_robustness_ensemble(
             existing_model_paths,
             BASE_MODEL_ARCH,
             num_classes,
             device,
             test_loader,
             dataset_name,
             RESULTS_DIR
         )
    else:
        # --- Run Ensemble Evaluation ---
        evaluate_robustness_ensemble(
            existing_model_paths,
            BASE_MODEL_ARCH,
            num_classes,
            device,
            test_loader,
            dataset_name,
            RESULTS_DIR
        )

    print("\nEnsemble Multi-Attack Script Finished.")
