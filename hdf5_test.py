import h5py
import mne
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the working directory to the location of your .hdf5 files
os.chdir("C:\\Users\\peters\\OneDrive - Universitaet Bern\\Dokumente\\hdf5")

cwd = os.getcwd()

# Open the .hdf5 file
with h5py.File('BA12ED - 3_REM 1.hdf5', 'r') as f:
    # Print the names of all datasets in the file
    for name in f.keys():
        print(name)
    # Extract the data and metadata
        
    channels = ['01_F3', '02_F4', '03_C3', '04_C4', '05_O2', '06_O1', '07_EOGH1', '08_EOGH2', '09_EMG', '10_EMG']
    eeg_data = {channel: f[channel][:] for channel in channels}
    sampling_rate = 265
    data = {channel: np.array(f[channel]) for channel in channels}

data_2d = np.vstack(list(data.values()))
info = mne.create_info(ch_names=list(data.keys()), sfreq=sampling_rate)
raw = mne.io.RawArray(data_2d, info)
raw.save('BA12ED - 3_REM 1_eeg.fif', overwrite=True)
raw = mne.io.read_raw_fif('BA12ED - 3_REM 1_eeg.fif', preload=True)

# Creating epochs
epoch_duration = 30.  # duration of each epoch in secondsh
samples_per_epoch = int(epoch_duration * sampling_rate)
data = raw.get_data()

# Check if the data can be split evenly into epochs
n_samples = data.shape[1]
remainder = n_samples % samples_per_epoch
if remainder != 0:
    # If not, pad the data with zeros so that it can be
    padding = samples_per_epoch - remainder
    data = np.pad(data, ((0, 0), (0, padding)))

# Split the raw data into consecutive epochs
epochs_data = np.array(np.split(data, np.arange(samples_per_epoch, len(data[0]), samples_per_epoch), axis=1))

# Create a new info object that matches the structure of the epochs_data
n_channels = epochs_data.shape[1]
channel_types = ['eeg'] * n_channels
channel_names = ['ch{}'.format(i) for i in range(n_channels)]
info = mne.create_info(channel_names, sampling_rate, channel_types)

# Create an epochs object from the epochs data
epochs = mne.EpochsArray(epochs_data, info)

# Define the scaling factors for each channel type
scalings = {'eeg': 1e-6, 'eog': 1e-6,}
# Setting the channel names
picks = ['01_F3', '02_F4', '03_C3', '04_C4', '05_O2', '06_O1', '07_EOGH1', '08_EOGH2', '09_EMG', '10_EMG']
channel_map = {old_name: new_name for old_name, new_name in zip(epochs.ch_names, picks)}
epochs.rename_channels(channel_map)

# Plot the epochs with the specified scaling
epochs.plot(n_epochs=1, n_channels=10, scalings=scalings)
plt.show(block=True)