import os
import torch
import numpy as np
import librosa
import musdb

def preprocess_musdb(
    root_dir='musdb18',
    save_path='full_spectrograms.pt',
    sample_rate=8192,
    n_fft=1024,
    hop_length=768,
    patch_frames=128
):
    print("Loading MUSDB dataset...")
    db = musdb.DB(root=root_dir, subsets=['train'])

    data = []

    for track_idx, track in enumerate(db.tracks):
        print(f"Processing track {track_idx+1}/{len(db.tracks)}: {track.name}")
        orig_sr = track.rate

        # Mono
        mixture = track.audio.mean(axis=1)
        vocals  = track.targets['vocals'].audio.mean(axis=1)

        # Resample
        mixture = librosa.resample(mixture, orig_sr=orig_sr, target_sr=sample_rate)
        vocals  = librosa.resample(vocals,  orig_sr=orig_sr, target_sr=sample_rate)

        # STFT magnitude
        X = np.abs(librosa.stft(mixture, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
        Y = np.abs(librosa.stft(vocals,  n_fft=n_fft, hop_length=hop_length, win_length=n_fft))

        # Keep first 512 freq bins
        X = X[:512, :]
        Y = Y[:512, :]

        # Normalize
        m = X.max() if X.max() > 0 else 1.0
        X /= m
        Y /= m

        # Pad if too short
        n_frames = X.shape[1]
        if n_frames < patch_frames:
            pad = patch_frames - n_frames
            X = np.pad(X, ((0, 0), (0, pad)), mode='constant')
            Y = np.pad(Y, ((0, 0), (0, pad)), mode='constant')

        # To tensor
        X = torch.from_numpy(X[None]).float()  # [1,512,T]
        Y = torch.from_numpy(Y[None]).float()

        data.append((X, Y))

    print(f"Saving {len(data)} tracks to {save_path}...")
    torch.save(data, save_path)
    print("Preprocessing done!")

if __name__ == '__main__':
    preprocess_musdb()
