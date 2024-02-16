import h5py
import mne
import matplotlib.pyplot as plt
import os
import numpy as np
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

# Change the directory and filename
# Set the working directory to the location of your .hdf5 files
os.chdir("C:\\Users\\peters\\OneDrive - Universitaet Bern\\Dokumente\\hdf5")
# Define the filename
filename = 'BE01AN - 3_REM 1'
#___________________________________________________________________________________#
# Open the .hdf5 file
with h5py.File(f'{filename}.hdf5', 'r') as f:
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
raw.save(f'{filename}_eeg.fif', overwrite=True)
raw = mne.io.read_raw_fif(f'{filename}_eeg.fif', preload=True)
cwd = os.getcwd()


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

#for i in range(len(epochs)):
    # Apply a Hann window function to each epoch
    #epochs._data[i] *= np.hanning(epochs._data.shape[2])

# Define the scaling factors for each channel type
scalings = {'eeg': 0.5e2}

# Setting the channel names 
picks = ['01_F3', '02_F4', '03_C3', '04_C4', '05_O2', '06_O1', '07_EOGH1', '08_EOGH2', '09_EMG', '10_EMG']
channel_map = {old_name: new_name for old_name, new_name in zip(epochs.ch_names, picks)}
epochs.rename_channels(channel_map)

# Setting the channel types
    # Get the names of 'misc' channels
misc_channels = [raw.ch_names[i] for i in mne.pick_types(raw.info, misc=True)]
    # Define a dictionary with new channel types
new_channel_types = {name: 'eeg' for name in misc_channels}
    # Change the channel types
raw.set_channel_types(new_channel_types)

# Calculate the signal length in samples
signal_length_samples = len(epochs.times)

# Set the filter length to be less than the signal length
filter_length = int(signal_length_samples * 0.9)  # 90% of the signal length

epochs.filter(l_freq=1, h_freq=30, filter_length=filter_length)
epochs.plot(n_epochs=1, n_channels=10, scalings=scalings, title='Epochs')
plt.show(block=True)

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the first channel of the first epoch
ax.plot(epochs.times, data[0, 0])

# Show the plot
plt.show(block=True)
