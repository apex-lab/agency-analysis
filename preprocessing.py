import numpy as np
import os.path as op
from pprint import pformat
from scipy.stats import lognorm
# EEG utilities
import mne
from mne.preprocessing import ICA, create_eog_epochs
from pyprep.prep_pipeline import PrepPipeline
from autoreject import get_rejection_threshold
# BIDS utilities
from mne_bids import BIDSPath, read_raw_bids
from util.io.bids import DataSink
from bids import BIDSLayout

# constants / config
BIDS_ROOT = 'bids_dataset'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
HIGHPASS = 1. # low cutoff for filter
LOWPASS = 30. # high cutoff for filter
TMIN = -.1
TMAX = .5

# gather our bearings
layout = BIDSLayout(BIDS_ROOT, derivatives = True)
subjects = layout.get_subjects()
subjects.sort()
already_done = layout.get_subjects(scope = 'preprocessing')

for i, sub in enumerate(subjects):

    if sub in already_done:
        continue # don't preprocess twice!

    np.random.seed(i)

    # grab the data
    bids_path = BIDSPath(
        root = BIDS_ROOT,
        subject = sub,
        task = 'agencyRT',
        datatype = 'eeg'
    )
    raw = read_raw_bids(bids_path, verbose = False)

    # get events of interest
    log = layout.get(subject = sub, suffix = 'events')[0].get_df()
    stim_trials = log[log.pressed_first == False]
    # remove outlier trials (i.e. where stimulation failed)
    stim_trials = stim_trials[stim_trials.rt < .6] # outside experimental bounds
    movement_lag = stim_trials.rt - stim_trials.latency
    params = lognorm.fit(movement_lag)
    lower = lognorm.ppf(.025, params[0], params[1], params[2])
    upper = lognorm.ppf(.975, params[0], params[1], params[2])
    outlier_idx = (movement_lag > upper) | (movement_lag < lower)
    stim_trials = stim_trials[~outlier_idx]
    # format for MNE
    stims = (stim_trials.onset + stim_trials.latency) * raw.info['sfreq']
    stim_samples = stims.to_numpy().astype(int)
    agency = stim_trials.agency.to_numpy()
    useless_col = np.zeros_like(agency)
    stim_events = np.stack([stim_samples, useless_col, agency], axis = 1)
    # and their corresponding visual cue times
    vep_samples = stim_trials.onset * raw.info['sfreq']
    vep_samples = vep_samples.to_numpy().astype(int)
    useless_col = np.zeros_like(vep_samples)
    vep_code = 2*np.ones_like(vep_samples)
    vep_events = np.stack([vep_samples, useless_col, vep_code], axis = 1)
    # combine
    events = np.concatenate([stim_events, vep_events])
    events = events[events[:, 0].argsort()]
    event_ids = {'agency': 1, 'non-agency': 0, 'cue': 2}

    # re-reference eye electrodes to become bipolar EOG
    raw.load_data()
    def reref(dat):
        dat[1,:] = (dat[1,:] - dat[0,:]) * -1
        return dat
    raw = raw.apply_function(
        reref,
        picks = ['leog', 'Fp2'],
        channel_wise = False
    )
    raw = raw.apply_function(
        reref,
        picks = ['reog', 'Fp1'],
        channel_wise = False
    )

    # fix stimuluation artifact before doing anything else
    raw.set_eeg_reference()
    raw = raw.notch_filter(
        raw.info['line_freq'] * np.arange(1, 4),
        method = 'spectrum_fit'
    )
    mne.preprocessing.fix_stim_artifact(
        raw,
        events = events[events[:, -1] != event_ids['cue']],
        tmin = -.005,
        tmax = .010
    )
    # more aggressively remove line noise now that stim artifact is interpolated
    # and we won't create more artifact by applying an FIR filter
    lfs = np.arange(
        raw.info['line_freq'],
        LOWPASS + raw.info['line_freq'],
        raw.info['line_freq']
    )
    raw = raw.notch_filter(lfs, method = 'fir', n_jobs = 5)
    raw, events = raw.resample(5000, events = events) # resample for PREP

    # run PREP pipeline (exclude bad chans, and re-reference)
    np.random.seed(i)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": [], # we already handled line noise earlier,
    }
    prep = PrepPipeline(
        raw,
        prep_params,
        raw.get_montage(),
        ransac = True,
        filter_kwargs = dict(n_jobs = 5),
        random_state = i
    )
    prep.fit()
    # recombine EOG channels
    raw = prep.raw_eeg.add_channels(
        [prep.raw_non_eeg],
        force_update_info = True
    )
    raw.info['bads'] = [] # already interpolated by PREP

    # filter data
    filt = raw.copy().filter(l_freq = HIGHPASS, h_freq = LOWPASS)
    # compute ICA components
    ica = ICA(n_components = 15, random_state = 0)
    ica.fit(filt, picks = 'eeg')
    # and exclude ICA components that are correlated with EOG
    eog_indices, eog_scores = ica.find_bads_eog(filt, threshold = 1.96)
    ica.exclude = eog_indices
    ica.apply(filt) # transforms in place
    # and finally remove EOG channels
    filt = filt.drop_channels(['leog', 'reog'])

    # extract epochs of interest
    epochs = mne.Epochs(
        filt,
        events,
        tmin = TMIN,
        tmax = TMAX,
        event_id = event_ids,
        baseline = None,
        preload = True
    )
    epochs = epochs[['agency', 'non-agency']]

    # and find optimal rejection threshold
    thres = get_rejection_threshold(
        epochs.copy().apply_baseline((TMIN, 0.))
    )

    # estimate visual evoked potential to cue with overlap correction
    evokeds = mne.stats.linear_regression_raw(
        filt,
        events, event_ids,
        tmin = TMIN,
        tmax = {'agency': TMAX, 'non-agency': TMAX, 'cue': .7 + TMAX},
        reject = thres # with optimal rejection threshold from above
    )

    # get average VEP, offset by stim latency per trial
    stim_starts = stim_trials.latency.to_numpy()
    stim_ends = stim_starts + TMAX
    stim_starts += TMIN
    vep = [
        evokeds['cue'].get_data(tmin = s, tmax = e)
        for s, e in zip(stim_starts, stim_ends)
    ]
    vep = np.stack(vep)

    # remove VEP from single-trial data
    if vep.shape[-1] < epochs._data.shape[-1]:
        epochs = epochs.crop(epochs.times[0], epochs.times[-2])
    epochs._data -= vep

    # now that that's done, clean up the single-trial data
    epochs = epochs.apply_baseline((TMIN, 0.)) # baseline correct
    epochs = epochs.drop_bad(reject = thres) # and drop artifacts
    epochs = epochs.resample(sfreq = 2 * LOWPASS)

    # save the cleaned data
    sink = DataSink(DERIV_ROOT, 'preprocessing')
    ev_fpath = sink.get_path(
        subject = sub,
        task = 'agencyRT',
        desc = 'reg',
        suffix = 'ave',
        extension = 'fif.gz'
    )
    mne.write_evokeds(
        ev_fpath,
        [evokeds[key] for key in evokeds],
        overwrite = True
    )
    fpath = sink.get_path(
        subject = sub,
        task = 'agencyRT',
        desc = 'clean',
        suffix = 'epo',
        extension = 'fif.gz'
    )
    epochs.save(fpath)

    # generate a preprocessing report
    report = mne.Report(verbose = True)
    report.parse_folder(
        op.dirname(fpath),
        pattern = '*epo.fif.gz',
        render_bem = False
    )
    fig_vep = evokeds['cue'].apply_baseline((TMIN, 0)).plot_joint(
        ts_args = dict(gfp = True),
        show = False
    )
    report.add_figure(
        fig_vep,
        title = 'Visual Evoked Response',
        section = 'evoked'
    )
    fig_agency = evokeds['agency'].apply_baseline((TMIN, 0)).plot_joint(
        ts_args = dict(gfp = True),
        show = False
    )
    report.add_figure(
        fig_agency,
        title = 'Evoked Response: Agency Condition',
        section = 'evoked'
    )
    fig_noagency = evokeds['non-agency'].apply_baseline((TMIN, 0)).plot_joint(
        ts_args = dict(gfp = True),
        show = False
    )
    report.add_figure(
        fig_noagency,
        title = 'Evoked Response: No Agency Condition',
        section = 'evoked'
    )
    diff = mne.combine_evoked(
        [evokeds['agency'], evokeds['non-agency']],
        weights = [1, -1]
    )
    fig_diff = diff.plot_joint(ts_args = dict(gfp = True), show = False)
    report.add_figure(
        fig_diff,
        title = 'Difference Wave (agency - no agency)',
        section = 'evoked'
    )
    if ica.exclude:
        fig_ica = ica.plot_components(ica.exclude, show = False)
        report.add_figure(
            fig_ica,
            title = 'Removed ICA Components',
            section = 'ICA'
        )
    bads = prep.noisy_channels_original
    html_lines = []
    for line in pformat(bads).splitlines():
        html_lines.append('<br/>%s' % line)
    html = '\n'.join(html_lines)
    report.add_html(
        html,
        title = 'Interpolated Channels'
    )
    crit = '<br/>threshold: {:0.2f} microvolts</br>'.format(thres['eeg'] * 1e6)
    report.add_html(
        crit,
        title = 'Peak-to-peak trial rejection threshold'
    )
    report.add_html(
        epochs.info._repr_html_(),
        title = 'Info'
    )
    report.save(op.join(sink.deriv_root, 'sub-%s.html'%sub), overwrite = True)
