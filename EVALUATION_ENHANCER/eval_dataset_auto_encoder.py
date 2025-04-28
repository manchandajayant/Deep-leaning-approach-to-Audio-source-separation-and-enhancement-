import os

import numpy as np
import torch
from enhancer import SpectrogramEnhancer  # Your GAN/VAE model

# === CONFIG ===
vae_test_folder = "GANAE_test"  # Folder containing all npz files
checkpoint_path = "enhancer_epoch100.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpectrogramEnhancer().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# === Metrics storage ===
mse_list = []
mae_list = []
spectral_convergence_list = []
lsd_list = []

npz_files = [f for f in os.listdir(vae_test_folder) if f.endswith(".npz")]
npz_files.sort()

for filename in npz_files:
    file_path = os.path.join(vae_test_folder, filename)
    print(f"Processing: {filename}")

    # === Load file
    data = np.load(file_path)
    input_spec = data["input"][:512, :]
    target_spec = data["target"][:512, :]
    phase = data["phase"][:512, :]
    scale = data["scale"]

    input_mag = input_spec * scale
    target_mag = target_spec * scale

    input_log = np.log1p(input_mag)
    input_tensor = torch.tensor(input_log).unsqueeze(0).unsqueeze(0).to(device).float()

    # === Run enhancer (GANCoder) inference
    with torch.no_grad():
        recon_log = model(input_tensor).squeeze().cpu().numpy()

    recon_log = np.clip(recon_log, a_min=-20, a_max=10)  # clip log values
    recon_mag = np.expm1(recon_log)

    #  Trim to same length
    min_T = min(recon_mag.shape[1], target_mag.shape[1])
    recon_mag_eval = recon_mag[:, :min_T]
    target_mag_eval = target_mag[:, :min_T]

    # Compute metrics
    mse = np.mean((recon_mag_eval - target_mag_eval) ** 2)
    mae = np.mean(np.abs(recon_mag_eval - target_mag_eval))

    num = np.linalg.norm(target_mag_eval - recon_mag_eval, ord="fro")
    den = np.linalg.norm(target_mag_eval, ord="fro")
    spectral_convergence = num / (den + 1e-8)

    log_diff = np.log10(target_mag_eval + 1e-8) - np.log10(recon_mag_eval + 1e-8)
    lsd = np.sqrt(np.mean(log_diff**2))

    # Save metrics
    mse_list.append(mse)
    mae_list.append(mae)
    spectral_convergence_list.append(spectral_convergence)
    lsd_list.append(lsd)

    print(f"✅ {filename} done | MSE: {mse:.6f} | MAE: {mae:.6f} | LSD: {lsd:.6f}")

# Print final average results
print("\n====== Final Results over VAE Test Set ======")
print(f"📊 Average MSE: {np.mean(mse_list):.6f}")
print(f"📊 Average MAE: {np.mean(mae_list):.6f}")
print(f"📊 Average Spectral Convergence: {np.mean(spectral_convergence_list):.6f}")
print(f"📊 Average Log-Spectral Distance (LSD): {np.mean(lsd_list):.6f}")
