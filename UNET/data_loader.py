import numpy as np
np.complex = complex    # for librosa compatibility
import torch
import random
import librosa
import musdb
from torch.utils.data import Dataset

class MusdbDataset(Dataset):
    def __init__(self, root_dir, subset='train',
                 sample_rate=8192, n_fft=1024,
                 hop_length=768, patch_frames=128):
        self.sample_rate  = sample_rate
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.patch_frames = patch_frames
        
        # Note: subsets must be a list
        self.db = musdb.DB(root=root_dir, subsets=[subset])
        # Only keep tracks that have 'vocals'
        self.tracks = [t for t in self.db.tracks if 'vocals' in t.targets]
        if not self.tracks:
            raise RuntimeError(f"No tracks found for subset={subset} in {root_dir}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        orig_sr = track.rate
        
        # track.audio is an array of shape (num_samples, 2)
        mixture    = track.audio.mean(axis=1)               # mono mix
        vocals     = track.targets['vocals'].audio.mean(axis=1)  # mono vocals
        
        # Resample to 8192 Hz
        mixture    = librosa.resample(mixture, orig_sr=orig_sr, target_sr=self.sample_rate)
        vocals     = librosa.resample(vocals,  orig_sr=orig_sr, target_sr=self.sample_rate)
        
        # Compute magnitude spectrograms
        X = np.abs(librosa.stft(mixture, n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                win_length=self.n_fft))
        Y = np.abs(librosa.stft(vocals,  n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                win_length=self.n_fft))
        
        # Keep first 512 freq bins
        X = X[:512, :]
        Y = Y[:512, :]
        
        # Normalize by max of mixture (avoid divide-by-zero)
        m = X.max() if X.max() > 0 else 1.0
        X /= m
        Y /= m
        
        # Pad if too short
        n_frames = X.shape[1]
        if n_frames < self.patch_frames:
            pad = self.patch_frames - n_frames
            X = np.pad(X, ((0,0),(0,pad)), mode='constant')
            Y = np.pad(Y, ((0,0),(0,pad)), mode='constant')
            n_frames = self.patch_frames
        
        # Random time patch
        start = random.randint(0, n_frames - self.patch_frames)
        Xp = X[:, start:start+self.patch_frames]
        Yp = Y[:, start:start+self.patch_frames]
        
        # To PyTorch tensors with shape [1,512,128]
        Xp = torch.from_numpy(Xp[None]).float()
        Yp = torch.from_numpy(Yp[None]).float()
        
        return Xp, Yp


from torch.utils.data import DataLoader

