import os
import torch
import torchaudio
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cadenza_system import CadenzaSystem
from deepafx_st.utils import system_summary
from pathlib import Path
from audiogram import Audiogram
from audiogram import Listener
from nalr import NALR
from compressor import Compressor
import json
import numpy as np
from numpy import ndarray

# class CustomDataset(Dataset):
#     def __init__(self, data_folder):
#         self.data_folder = data_folder
#         self.file_list = os.listdir(data_folder)

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_path = os.path.join(self.data_folder, self.file_list[idx])
#         waveform, sample_rate = torchaudio.load(file_path)
        
#         # You can apply additional preprocessing here if needed
#         # For example, resampling, normalizing, etc.

#         # Assuming the folder structure is organized in a way that the label
#         # is encoded in the filename or folder structure, you can extract it.
#         # For example, if the file is named "label1_file1.wav", you can do:
#         label = int(self.file_list[idx].split('_')[0].replace('label', ''))

#         return waveform, label

def apply_ha(
    enhancer: NALR,
    compressor: Compressor,
    signal: ndarray,
    audiogram: Audiogram,
    apply_compressor: bool = False,
) -> np.ndarray:
    """
    Apply NAL-R prescription hearing aid to a signal.

    Args:
        enhancer (NALR): A NALR object that enhances the signal.
        compressor (Compressor | None): A Compressor object that compresses the signal.
        signal (ndarray): An ndarray representing the audio signal.
        audiogram (Audiogram): An Audiogram object representing the listener's
            audiogram.
        apply_compressor (bool): Whether to apply the compressor.

    Returns:
        An ndarray representing the processed signal.
    """
    nalr_fir, _ = enhancer.build(audiogram)
    proc_signal = enhancer.apply(nalr_fir, signal)
    if apply_compressor:
        if compressor is None:
            raise ValueError("Compressor must be provided to apply compressor.")

        proc_signal, _, _ = compressor.process(proc_signal)
    return proc_signal

def process_remix_for_listener(
    signal: ndarray,
    enhancer: NALR,
    compressor: Compressor,
    listener: Listener,
    apply_compressor: bool = False,
) -> ndarray:
    """Process the stems from sources.

    Args:
        stems (dict) : Dictionary of stems
        sample_rate (float) : Sample rate of the signal
        enhancer (NALR) : NAL-R prescription hearing aid
        compressor (Compressor) : Compressor
        listener: Listener object
        apply_compressor (bool) : Whether to apply the compressor
    Returns:
        ndarray: Processed signal.
    """

    left_output = apply_ha(enhancer, compressor, signal[0, :], listener.audiogram_left, apply_compressor)
    right_output = apply_ha(enhancer, compressor, signal[1, :], listener.audiogram_right, apply_compressor)

    return np.stack([left_output, right_output], axis=0)

def make_scene_listener_list(scenes_listeners: dict, small_test: bool = False) -> list:
    """Make the list of scene-listener pairing to process

    Args:
        scenes_listeners (dict): Dictionary of scenes and listeners.
        small_test (bool): Whether to use a small test set.

    Returns:
        list: List of scene-listener pairings.

    """
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]

    # Can define a standard 'small_test' with just 1/15 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::15]

    return scene_listener_pairs

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # create the System
    system = CadenzaSystem()

    # print details about the model
    system_summary(system)

    # Example usage
    clean_folder = '/Users/emilykuo/Desktop/cadenza_data/clean'
    target_folder = '/Users/emilykuo/Desktop/cadenza_data/target'

    metadata_path = "/Users/emilykuo/Desktop/cadenza_data/metadata/"

    with Path(metadata_path + "gains.json").open("r", encoding="utf-8") as file:
        gains = json.load(file)

    with Path(metadata_path + "scenes.train.json").open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    with Path(metadata_path + "scene_listeners.train.json").open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    with Path(metadata_path + "at_mic_music.train.json").open("r", encoding="utf-8") as file:
        songs = json.load(file)

    enhancer = NALR(nfir=220, sample_rate=44100)
    # TODO: make the compressor differentiable
    compressor = Compressor(threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064)

    scene_listener_pairs = make_scene_listener_list(scenes_listeners)

    listener_dict = Listener.load_listener_dict(metadata_path + "listeners.train.json")

    num_scenes = len(scene_listener_pairs)
    num_epochs = 1
    desired_duration = 10
    desired_samples = int(desired_duration * 44100)

    for epoch in range(num_epochs):
        for idx, scene_listener_pair in enumerate(scene_listener_pairs, 1):
            scene_id, listener_id = scene_listener_pair

            scene = scenes[scene_id]
            song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"

            print(f"[{idx:03d}/{num_scenes:03d}] ")
            print(f"Processing {scene_id}: {song_name} for listener {listener_id}")

            logger.info(
                f"[{idx:03d}/{num_scenes:03d}] "
                f"Processing {scene_id}: {song_name} for listener {listener_id}"
            )
            # Get the listener's audiogram
            listener = listener_dict[listener_id]

            target_filepath = os.path.join(target_folder, song_name)
            target_waveform, target_sample_rate = torchaudio.load(target_filepath + ".wav")

            last_hyphen_index = song_name.rfind('-')
            # Extract the string before the last hyphen
            clean_song_name = song_name[:last_hyphen_index]
            clean_filepath = os.path.join(clean_folder + clean_song_name)
            clean_waveform, clean_sample_rate = torchaudio.load(target_filepath + ".wav")

            assert target_sample_rate == clean_sample_rate


            start_sample = torch.randint(0, target_waveform.shape[1] - desired_samples, ())

            # Extract the desired duration of audio from both channels
            selected_target_waveform = target_waveform[:, start_sample:start_sample + desired_samples]
            selected_clean_waveform = clean_waveform[:, start_sample:start_sample + desired_samples]

            selected_enhanced_waveform = process_remix_for_listener(
                signal=selected_clean_waveform,
                enhancer=enhancer,
                compressor=compressor,
                listener=listener,
                apply_compressor=False,
            )
            selected_enhanced_waveform = torch.from_numpy(selected_enhanced_waveform)[:, :desired_samples]

            gain_params = list(gains[scene["gain"]].values())
            print(gain_params)

            
            
