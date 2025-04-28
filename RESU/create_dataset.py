# Load Tracks
import numpy as np

np.complex = complex  # for librosa compatibility
import os

import librosa
import musdb
import soundfile as sf
import torch
from IPython.display import Audio, display

from UNET.model import UNetVocalSeparator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetVocalSeparator(base_channels=32).to(DEVICE)
model.load_state_dict(torch.load("unet_vocal_epoch200.pt", map_location=DEVICE))
model.eval()

sr_unet = 8192
sr_full = 44100
n_fft = 1024
hop = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_channels = 32
pad_multiple = 16

save_path = "vae_prepped"
os.makedirs(save_path, exist_ok=True)

tracks = musdb.DB(root="UNET/musdb18", subsets="train").tracks

pad_multiple_unet = 64
pad_multiple_vae = 16

for track in tracks:
    print(f"Processing: {track.name}")

    # 1. Mono mixture and vocals
    mix = track.audio.mean(axis=1)
    gt_vocals = track.targets["vocals"].audio.mean(axis=1)

    # 2. Resample to 8192 Hz
    mix_rs = librosa.resample(mix, orig_sr=track.rate, target_sr=sr_unet)
    gt_rs = librosa.resample(gt_vocals, orig_sr=track.rate, target_sr=sr_unet)

    # 3. STFT and magnitude/phase
    S_mix = librosa.stft(mix_rs, n_fft=n_fft, hop_length=hop)
    mag_mix, phase_mix = np.abs(S_mix), np.angle(S_mix)
    mag_mix = mag_mix[:512, :]
    phase_mix = phase_mix[:512, :]

    # 4. Normalize and pad to 64 frames (for UNet input)
    scale = mag_mix.max() if mag_mix.max() > 0 else 1.0
    mag_norm = mag_mix / scale

    T = mag_norm.shape[1]
    pad = (pad_multiple_unet - (T % pad_multiple_unet)) % pad_multiple_unet

    mag_padded = np.pad(mag_norm, ((0, 0), (0, pad)), mode="constant")
    phase_padded = np.pad(phase_mix, ((0, 0), (0, pad)), mode="constant")
    mag_input = mag_padded[None, None]  # [1,1,512,T]

    # 5. Inference
    with torch.no_grad():
        mask = model(torch.from_numpy(mag_input).float().to(DEVICE))
    mask = mask.cpu().numpy()[0, 0, :, :T]  # Unpad output to match original T

    # 6. Masked magnitude
    vocal_mag = mask * mag_mix

    # 7. Reconstruct waveform
    full_spec = np.pad(
        vocal_mag * np.exp(1j * phase_mix),
        ((0, n_fft // 2 + 1 - 512), (0, 0)),
        mode="constant",
    )
    pred_audio = librosa.istft(full_spec, hop_length=hop, win_length=n_fft)

    # 8. Resample both prediction and GT to 44.1kHz
    pred_44 = librosa.resample(pred_audio, orig_sr=sr_unet, target_sr=sr_full)
    gt_44 = librosa.resample(gt_rs, orig_sr=sr_unet, target_sr=sr_full)

    # 9. Final STFTs for VAE input
    S_pred = librosa.stft(pred_44, n_fft=n_fft, hop_length=hop)
    S_gt = librosa.stft(gt_44, n_fft=n_fft, hop_length=hop)

    mag_pred, phase_pred = np.abs(S_pred), np.angle(S_pred)
    mag_gt, phase_gt = np.abs(S_gt), np.angle(S_gt)

    # 10. Pad to multiple of 16 frames (for VAE input)
    T_final = mag_pred.shape[1]
    pad_final = (pad_multiple_vae - (T_final % pad_multiple_vae)) % pad_multiple_vae

    mag_pred = np.pad(mag_pred, ((0, 0), (0, pad_final)), mode="constant")
    mag_gt = np.pad(mag_gt, ((0, 0), (0, pad_final)), mode="constant")
    phase_pred = np.pad(phase_pred, ((0, 0), (0, pad_final)), mode="constant")
    phase_gt = np.pad(phase_gt, ((0, 0), (0, pad_final)), mode="constant")

    # 11. Save
    out_path = f"{save_path}/{track.name}_vae.npz"
    np.savez_compressed(
        out_path,
        input=mag_pred.astype(np.float32),
        target=mag_gt.astype(np.float32),
        phase=phase_pred.astype(np.float32),
        phase_gt=phase_gt.astype(np.float32),
        scale=np.float32(scale),
    )

    print(f"Saved: {out_path}")
