import os
import torch
from torch.utils.data import Dataset
import torchaudio
import re
import json
from pathlib import Path
from audiogram import Listener

class CadenzaDataset(Dataset):
    def __init__(self, root_dir, subset='train', duration=10, fs=44100):
        self.root_dir = root_dir
        self.subset = subset
        self.duration = duration
        self.fs = fs
        self.num_samples = int(self.duration * fs)

        self.original_folder_path = os.path.join(self.root_dir, "audio", "at_mic_music")
        self.reference_folder_path = os.path.join(self.root_dir, self.subset, "reference")
        self.metadata_folder_path = os.path.join(self.root_dir, "metadata")

        with Path(os.path.join(self.metadata_folder_path, "gains.json")).open("r", encoding="utf-8") as file:
            self.gains = json.load(file)

        with Path(os.path.join(self.metadata_folder_path, "scenes." + subset + ".json")).open("r", encoding="utf-8") as file:
            self.scenes = json.load(file)

        with Path(os.path.join(self.metadata_folder_path, "scene_listeners." + subset + ".json")).open("r", encoding="utf-8") as file:
            self.scenes_listeners = json.load(file)

        with Path(os.path.join(self.metadata_folder_path, "at_mic_music." + subset + ".json")).open("r", encoding="utf-8") as file:
            self.songs = json.load(file)

        self.listener_dict = Listener.load_listener_dict(os.path.join(self.metadata_folder_path, "listeners." + subset + ".json"))

        self.segments = self._extract_segments()

    def _extract_segments(self):
        segments = []
        reference_file_list = os.listdir(self.reference_folder_path)
        
        for reference_filename in reference_file_list:
            if reference_filename.endswith(".flac"):

                # Using regular expressions to extract the desired parts
                match = re.match(r"scene_(\d+)_L(\d+)_", reference_filename)

                if match:
                    scene_id = "scene_" + match.group(1)
                    listener_id = "L" + match.group(2)
                else:
                    raise ValueError("Pattern not found in the filename.")
                    
                scene = self.scenes[scene_id]
                song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"
                gain_values = list(self.gains[scene["gain"]].values())

                original_filepath = os.path.join(self.original_folder_path, self.songs[song_name]["Path"], "mixture.wav")
                original_waveform, sample_rate = torchaudio.load(original_filepath, normalize=True)
                self._check_validity(original_filepath, original_waveform, sample_rate)

                reference_filepath = os.path.join(self.reference_folder_path, reference_filename)
                reference_waveform, sample_rate = torchaudio.load(reference_filepath, normalize=True)
                self._check_validity(reference_filepath, reference_waveform, sample_rate)

                channel_count = reference_waveform.size(0)
                sample_count = reference_waveform.size(1)
                
                # Extract segments and append to the list
                for start in range(0, sample_count - self.num_samples + 1, self.num_samples):
                    for ch in range(channel_count):
                        original_audio_excerpt = original_waveform[ch, start:start + self.num_samples]
                        reference_audio_excerpt = reference_waveform[ch, start:start + self.num_samples]
                        gain_tensor = torch.tensor(gain_values, dtype=torch.float, requires_grad=True)
                        segments.append((original_audio_excerpt, reference_audio_excerpt, gain_tensor))

        return segments

    def _check_validity(self, filename, waveform, sample_rate):
        # Ensure the audio file matches the intended sample rate
        if sample_rate != self.fs:
            raise ValueError(f"The audio file {filename} has a {sample_rate} Hz sample rate.")
        
        # Ensure the audio file is stereo
        if waveform.size(0) != 2:
            raise ValueError(f"The audio file {filename} is not stereo.")

        if waveform.size(1) < self.num_samples:
            raise ValueError(f"The audio file {filename} is too short for the desired duration.")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]