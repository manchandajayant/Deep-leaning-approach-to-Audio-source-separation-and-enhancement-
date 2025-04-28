import os

import librosa
import musdb
import numpy as np
import torch

from UNET.model import UNetVocalSeparator

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained UNet model
model = UNetVocalSeparator(base_channels=32).to(DEVICE)
model.load_state_dict(torch.load("unet_vocal_epoch200.pt", map_location=DEVICE))
model.eval()

# MUSDB setup
musdb_root = "UNET/musdb18"  # <- CHANGE this to where MUSDB is
tracks = musdb.DB(root=musdb_root, subsets="test").tracks

# STFT and resampling configs
n_fft = 1024
hop = 768
sr_unet = 8192
sr_full = 44100
pad_multiple_unet = 64
pad_multiple_vae = 16

# Where to save the output
save_path = "GANAE_test"
os.makedirs(save_path, exist_ok=True)

# === PROCESS ALL TEST TRACKS ===
for track in tracks:
    print(f"Processing: {track.name}")

    # === 1. Load mono mixture ===
    mix = track.audio.mean(axis=1)

    # === 2. Resample to 8192 Hz ===
    mix_rs = librosa.resample(mix, orig_sr=track.rate, target_sr=sr_unet)

    # === 3. STFT and mag/phase
    S_mix = librosa.stft(mix_rs, n_fft=n_fft, hop_length=hop)
    mag_mix, phase_mix = np.abs(S_mix), np.angle(S_mix)
    mag_mix = mag_mix[:512, :]
    phase_mix = phase_mix[:512, :]

    # === 4. Normalize and pad to multiple of 64
    scale = mag_mix.max() if mag_mix.max() > 0 else 1.0
    mag_norm = mag_mix / scale

    T = mag_norm.shape[1]
    pad_unet = (pad_multiple_unet - (T % pad_multiple_unet)) % pad_multiple_unet

    mag_padded = np.pad(mag_norm, ((0, 0), (0, pad_unet)), mode="constant")
    phase_padded = np.pad(phase_mix, ((0, 0), (0, pad_unet)), mode="constant")

    mag_input = mag_padded[None, None]  # [1,1,512,T]

    # === 5. UNET inference
    with torch.no_grad():
        mask = model(torch.from_numpy(mag_input).float().to(DEVICE))
    mask = mask.cpu().numpy()[0, 0, :, :T]  # remove batch dim and unpad

    # === 6. Masked magnitude
    vocal_mag = mask * mag_mix

    # === 7. Reconstruct separated vocals waveform
    full_spec = np.pad(
        vocal_mag * np.exp(1j * phase_mix),
        ((0, n_fft // 2 + 1 - 512), (0, 0)),
        mode="constant",
    )
    pred_audio = librosa.istft(full_spec, hop_length=hop, win_length=n_fft)

    # === 8. Resample predicted and input back to 44.1kHz
    pred_44 = librosa.resample(pred_audio, orig_sr=sr_unet, target_sr=sr_full)
    mix_44 = librosa.resample(mix_rs, orig_sr=sr_unet, target_sr=sr_full)

    # === 9. STFT again at 44.1kHz for VAE input/target
    S_pred = librosa.stft(pred_44, n_fft=n_fft, hop_length=hop)
    S_mix = librosa.stft(mix_44, n_fft=n_fft, hop_length=hop)

    mag_pred, phase_pred = np.abs(S_pred), np.angle(S_pred)
    mag_gt, phase_gt = np.abs(S_mix), np.angle(S_mix)

    # === 10. Pad to multiple of 16
    T_final = mag_pred.shape[1]
    pad_vae = (pad_multiple_vae - (T_final % pad_multiple_vae)) % pad_multiple_vae

    mag_pred = np.pad(mag_pred, ((0, 0), (0, pad_vae)), mode="constant")
    mag_gt = np.pad(mag_gt, ((0, 0), (0, pad_vae)), mode="constant")
    phase_pred = np.pad(phase_pred, ((0, 0), (0, pad_vae)), mode="constant")
    phase_gt = np.pad(phase_gt, ((0, 0), (0, pad_vae)), mode="constant")

    # === 11. Save .npz file
    save_filename = os.path.join(save_path, f"{track.name}_vae.npz")
    np.savez_compressed(
        save_filename,
        input=mag_pred.astype(np.float32),
        target=mag_gt.astype(np.float32),
        phase=phase_pred.astype(np.float32),
        phase_gt=phase_gt.astype(np.float32),
        scale=np.float32(scale),
    )

    print(f"✅ Saved {save_filename}")

print("\n✅ Done saving all MUSDB test tracks to npz!")
