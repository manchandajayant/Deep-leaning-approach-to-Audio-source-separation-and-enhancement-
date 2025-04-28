# THIS FILE CREATES A FILE FOR INFERENCE TO THE MODEL
import os

import librosa
import numpy as np
import torch
from UNET.model import UNetVocalSeparator


def create_npz(input_mix_path, save_npz_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetVocalSeparator(base_channels=32).to(DEVICE)
    model.load_state_dict(
        torch.load("UNET/unet_vocal_epoch200.pt", map_location=DEVICE)
    )
    model.eval()

    n_fft = 1024
    hop = 768
    sr_unet = 8192
    sr_full = 44100
    pad_multiple_unet = 64
    pad_multiple_vae = 16

    # LOAD AUDIO
    mix, sr_mix = librosa.load(input_mix_path, sr=None, mono=True)

    mix_rs = librosa.resample(mix, orig_sr=sr_mix, target_sr=sr_unet)

    # STFT and mag/phase
    S_mix = librosa.stft(mix_rs, n_fft=n_fft, hop_length=hop)
    mag_mix, phase_mix = np.abs(S_mix), np.angle(S_mix)
    mag_mix = mag_mix[:512, :]
    phase_mix = phase_mix[:512, :]

    #  Normalize and pad to multiple of 64
    scale = mag_mix.max() if mag_mix.max() > 0 else 1.0
    mag_norm = mag_mix / scale

    T = mag_norm.shape[1]
    pad_unet = (pad_multiple_unet - (T % pad_multiple_unet)) % pad_multiple_unet

    mag_padded = np.pad(mag_norm, ((0, 0), (0, pad_unet)), mode="constant")
    phase_padded = np.pad(phase_mix, ((0, 0), (0, pad_unet)), mode="constant")

    mag_input = mag_padded[None, None]  # [1,1,512,T]

    # UNET inference
    with torch.no_grad():
        mask = model(torch.from_numpy(mag_input).float().to(DEVICE))
    mask = mask.cpu().numpy()[0, 0, :, :T]  # remove batch dim and unpad

    #  Masked magnitude
    vocal_mag = mask * mag_mix

    #  Reconstruct separated vocals waveform
    full_spec = np.pad(
        vocal_mag * np.exp(1j * phase_mix),
        ((0, n_fft // 2 + 1 - 512), (0, 0)),
        mode="constant",
    )
    pred_audio = librosa.istft(full_spec, hop_length=hop, win_length=n_fft)

    #  Resample predicted and input back to 44.1kHz
    pred_44 = librosa.resample(pred_audio, orig_sr=sr_unet, target_sr=sr_full)
    mix_44 = librosa.resample(mix_rs, orig_sr=sr_unet, target_sr=sr_full)

    # STFT again at 44.1kHz for VAE input/target
    S_pred = librosa.stft(pred_44, n_fft=n_fft, hop_length=hop)
    S_mix = librosa.stft(mix_44, n_fft=n_fft, hop_length=hop)

    mag_pred, phase_pred = np.abs(S_pred), np.angle(S_pred)
    mag_gt, phase_gt = np.abs(S_mix), np.angle(S_mix)

    # Pad to multiple of 16
    T_final = mag_pred.shape[1]
    pad_vae = (pad_multiple_vae - (T_final % pad_multiple_vae)) % pad_multiple_vae

    mag_pred = np.pad(mag_pred, ((0, 0), (0, pad_vae)), mode="constant")
    mag_gt = np.pad(mag_gt, ((0, 0), (0, pad_vae)), mode="constant")
    phase_pred = np.pad(phase_pred, ((0, 0), (0, pad_vae)), mode="constant")
    phase_gt = np.pad(phase_gt, ((0, 0), (0, pad_vae)), mode="constant")

    # Assign to match your eval variables
    recon_mag = mag_pred
    target_mag = mag_gt

    # SAVE
    os.makedirs(os.path.dirname(save_npz_path), exist_ok=True)

    np.savez_compressed(
        save_npz_path,
        input=recon_mag.astype(np.float32),
        target=target_mag.astype(np.float32),
        phase=phase_pred.astype(np.float32),
        phase_gt=phase_gt.astype(np.float32),
        scale=np.float32(scale),
    )

    print(f"✅ Saved NPZ for VAE: {save_npz_path}")
