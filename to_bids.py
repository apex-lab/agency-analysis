from mne_bids import (
    BIDSPath,
    write_raw_bids,
    get_anonymization_daysback,
    update_sidecar_json
)
import mne
from bids import BIDSLayout
from philistine.mne import write_raw_brainvision as write_bv

import pandas as pd
import numpy as np
import os
import re
import json
import tempfile

DATA_DIR = 'source' # where our data currently lives
BIDS_DIR = 'bids_dataset' # where we want it to live

try: # if bids directory has already been made
    layout = BIDSLayout(BIDS_DIR)
    finished_subjects = layout.get_subjects()
    finished_subjects = [int(s) for s in finished_subjects]
except:
    finished_subjects = [] # no subjects have been bidsified yet

eeg_fnames = os.listdir(os.path.join(DATA_DIR, 'eeg'))
eeg_fnames = [f for f in eeg_fnames if '.vhdr' in f] # filter for .vhdr files

loc_fnames = os.listdir(os.path.join(DATA_DIR, 'captrak'))
loc_fnames = [f for f in loc_fnames if '.bvct' in f]
loc_fnames = {int(re.findall('(\d+).bvct', f)[0]): f for f in loc_fnames}

log_fnames = os.listdir(os.path.join(DATA_DIR, 'logs'))
log_fnames = {int(re.findall('(\d+).tsv', f)[0]): f for f in log_fnames}

for f in eeg_fnames:

    # get relevant filepaths
    sub = int(re.findall('sub-(\d+).vhdr', f)[0])
    eeg_f = os.path.join(DATA_DIR, 'eeg', f)
    log_f = os.path.join(DATA_DIR, 'logs', log_fnames[sub])
    loc_f = os.path.join(DATA_DIR, 'captrak', loc_fnames[sub])

    if sub in finished_subjects or sub in [6, 24]:
        continue # move on to next iteration of loop

    # read EEG file
    raw = mne.io.read_raw_brainvision(eeg_f, preload = True)

    # rename EEG channels to 10-20 positions using Brain Vision provided layout file
    layout = mne.channels.read_custom_montage(os.path.join(DATA_DIR, 'AP-96.bvef'))
    mapping = {'Ch%s'%i: layout.ch_names[i] for i in range(len(layout.ch_names))}
    mapping = {key: value for key, value in mapping.items() if key in raw.ch_names}
    raw = raw.rename_channels(mapping)
    raw = mne.add_reference_channels(raw, 'Cz')

    # rename non-EEG channels
    raw = raw.rename_channels({'AF7': 'leog', 'AF8': 'reog'})
    raw = raw.set_channel_types({'leog': 'eog', 'reog': 'eog', 'photo': 'stim'})

    events, event_ids = mne.events_from_annotations(raw)

    log = pd.read_csv(log_f, sep = '\t')
    sample = events[events[:, 2] == 1, 0]
    onsets = sample/raw.info['sfreq']
    log.rt = log.rt * 1e-3
    log.insert(0, 'duration', np.zeros_like(0.))
    log.insert(0, 'onset', onsets)

    # correct timing with more precise photo sensor derived events
    raw = raw.apply_function(lambda x: (x > 1).astype(float), picks = 'photo')
    photo_events = mne.find_events(raw, output = 'step', initial_event = True)
    photo_starts = photo_events[photo_events[:, 2] == 1]
    photo_stops = photo_events[photo_events[:, 2] == 0]
    photo_dur = (photo_stops[:, 0] - photo_starts[:, 0])/raw.info['sfreq']
    trial_starts = photo_starts[:, 0]/raw.info['sfreq']
    idx1 = trial_starts >= log.onset[0] - 1
    trial_starts = trial_starts[idx1]
    idx2 = trial_starts <= log.onset[log.shape[0] - 1] + 1
    trial_starts = trial_starts[idx2]
    offsets = trial_starts - log.onset
    log.onset = log.onset + offsets
    log.rt = log.rt - offsets
    log.duration = photo_dur[idx1][idx2]

    # and add true stimulation latency
    log = log.replace(-1.0, np.nan)
    stim_times = events[events[:, 2] == 2, 0]/raw.info['sfreq']
    stim_latencies = stim_times - log.onset[log.trial_type == 'stimulation']
    # correct known trigger offset for stimulation (1.23 ms)
    log.latency[log.trial_type == 'stimulation'] = stim_latencies + .00123

    # pybv is ridiculously memory heavy so return to disk manually
    temp_dir = tempfile.TemporaryDirectory()
    temp_f = os.path.join(temp_dir.name, 'raw.vhdr')
    write_bv(raw, temp_f, events = False)
    raw = mne.io.read_raw_brainvision(temp_f, preload = False)
    raw = raw.set_channel_types({'leog': 'eog', 'reog': 'eog'})
    # read electrode positions
    dig = mne.channels.read_dig_captrak(loc_f)
    raw = raw.set_montage(dig)
    raw.info['line_freq'] = 60

    # write data to BIDS directory
    bids_path = BIDSPath(
        subject = '%02d'%sub,
        task = 'agencyRT',
        datatype = 'eeg',
        root = BIDS_DIR
    )
    saved_at = write_raw_bids(
        raw, bids_path = bids_path,
        overwrite = True
    )
    temp_dir.cleanup()

    # save new events file
    events_fpath = str(saved_at).replace('_eeg.vhdr', '_events.tsv')
    log.to_csv(events_fpath, sep = '\t', index = False, na_rep = 'n/a')

    # update sidecar json with extra fields
    json_fpath = str(saved_at).replace('vhdr', 'json')
    with open(json_fpath, "r") as f:
        desc = json.load(f)
    desc['EEGReference'] = 'Cz'
    desc['EEGGround'] = 'Fpz'
    desc['EEGPlacementScheme'] = 'extended 10-20'
    with open(json_fpath, "w") as f:
        json.dump(desc, f, indent = 4)
