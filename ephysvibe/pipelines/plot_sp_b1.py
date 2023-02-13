# plot spiking activity task b1
import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
from ..trials import select_trials
from ..trials.spikes import firing_rate, sp_constants
from ..task import def_task
from ..spike_sorting import config
from ..task import task_constants
import warnings
from matplotlib import pyplot as plt
from ..structures.trials_data import TrialsData

warnings.filterwarnings("ignore")


def main(
    filepath: Path,
    output_dir: Path,
    in_out: int,
    e_align: int,
    cgroup: str,
):
    """Compute and plot firing rate during task b1.

    Args:
        filepath (Path): path to the sorted file (.npy).
        output_dir (Path): output directory.
        in_out (int): 1 for trials with stimuli in, -1 out.
        e_align (int): event to which align the spikes.
        cgroup (str): "good" for individual units, "mua" for multiunits.
    """
    x_lim_max = 4
    x_lim_min = -0.7
    s_path = os.path.normpath(filepath).split(os.sep)[-1][:-4]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path])
    log_output = output_dir + "/" + s_path + "_plot_sp_b1.log"
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(
        handlers=[logging.FileHandler(log_output), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s | %(message)s ",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    # check if filepath exist
    if not os.path.exists(filepath):
        raise FileExistsError
    logging.info("-- Start --")
    # file = np.load(filepath, allow_pickle=True).item(0)
    # sp, bhv = file["sp_data"], file["bhv"]
    data = TrialsData.from_python_hdf5(filepath)
    trial_idx = np.where(np.logical_and(data.trial_error == 0, data.block == 1))[0]
    # Select trials and create task frame
    # trial_idx = select_trials.select_trials_block(sp, n_block=1)
    # trial_idx = select_trials.select_correct_trials(bhv, trial_idx)
    task = def_task.create_task_frame(trial_idx, bhv, task_constants.SAMPLES_COND)
    fig_task, _ = def_task.info_task(task)
    neurons = np.where(sp["clustersgroup"] == cgroup)[0]
    logging.info("Number of clusters: %d" % len(sp["clustersgroup"]))
    logging.info("Number of %s units: %d" % (cgroup, len(neurons)))
    logging.info("in_out: %d" % in_out)
    # logging.info("e_align: %s" % list(task_constants.EVENTS_B1.keys())[e_align])
    # define kernel for convolution
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )
    # select the trials

    # sample_id = task[(task["in_out"] == in_out)]["sample_id"].values
    # samples = np.sort(np.unique(sample_id))
    # target_trials_idx = task[(task["in_out"] == in_out)]["idx_trial"].values

    samples = np.sort(np.unique(task["sample_id"].values))
    target_trials_idx = task["idx_trial"].values
    condition = {-1: "out", 1: "in"}
    # plot fr for each neuron
    for i_neuron, neuron in enumerate(neurons):
        ev_ts = firing_rate.select_events_timestamps(
            sp, target_trials_idx, task_constants.EVENTS_B1
        )  # select events timestamps for all trials
        neuron_trials = firing_rate.align_neuron_spikes(
            target_trials_idx, sp, neuron, ev_ts[:, 0]
        )  # align sp with start trial
        shift_ev_ts = np.floor(
            ((ev_ts.T - ev_ts.T[0]).T) / config.DOWNSAMPLE
        )  # aling events with start trial
        # !
        trials_sp = firing_rate.sp_from_timestamp_to_binary(
            neuron_trials, config.DOWNSAMPLE
        )  # create arrays where if sp 1, else 0, at each timestamp
        _, max_shift, events_shift = firing_rate.reshape_sp_list(
            trials_sp, event_timestamps=shift_ev_ts, align_event=e_align
        )  # add zeros so each array (trial) has the same shape
        events_shift = (np.mean(events_shift, axis=0) - max_shift) / (
            config.FS / config.DOWNSAMPLE
        )  # ! graphs in ms
        conv = firing_rate.compute_conv_fr(
            neuron_trials, kernel, (config.FS / config.DOWNSAMPLE), config.DOWNSAMPLE
        )
        all_trials_fr, _, _ = firing_rate.aligne_conv_fr(
            conv=conv, event_timestamps=shift_ev_ts, align_event=e_align
        )  # aligne conv with e_align
        neuron_trials_shift = (
            firing_rate.align_neuron_spikes(target_trials_idx, sp, neuron, ev_ts[:, 2])
            / config.FS
        )  # align sp with stim onset
        # Iterate by sample and condition
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharey=True)
        for i_ax, cond in enumerate(condition.keys()):
            trials_conv_fr, all_mask = [], []
            sample_id = task[(task["in_out"] == cond)]["sample_id"].values
            for i_sample in samples:
                mask_sample = np.logical_and(
                    task["sample_id"].values == i_sample, task["in_out"] == cond
                )
                all_mask.append(sample_id == i_sample)
                trials_conv_fr.append(np.mean(all_trials_fr[mask_sample, :], axis=0))
            trials_conv_fr = np.array(trials_conv_fr)
            trials_time = (np.arange(len(trials_conv_fr[0])) - max_shift) / (
                config.FS / config.DOWNSAMPLE
            )
            # num_trials = len(neuron_trials)

            firing_rate.plot_b1(
                ax[i_ax],
                samples,
                trials_conv_fr,
                trials_time,
                neuron_trials_shift[task["in_out"] == cond],
                events_shift,
                cond,
                x_lim_min,
                x_lim_max,
                all_mask,
            )
        ax[1].legend(fontsize=9)
        # fig.legend()
        fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
        fig.text(
            0.10,
            0.01,
            s="%s - Aligned with %s"
            % (s_path[:10], list(task_constants.EVENTS_B1.keys())[e_align]),
            horizontalalignment="center",
            verticalalignment="center",
        )
        fig.suptitle("Neuron (%s) %d" % (cgroup, i_neuron + 1), x=0.10)

        if output_dir:

            logging.info("Saving figure, neuron: %d" % (i_neuron + 1))
            fig.savefig(
                "/".join(
                    [os.path.normpath(output_dir)]
                    + [
                        s_path
                        + "_n"
                        + str(i_neuron + 1)
                        + "_"
                        + cgroup
                        + "_"
                        + condition[in_out]
                        + "_b1.jpg"
                    ]
                )
            )
    fig_task.savefig(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_info_task_b1.jpg"])
    )
    logging.info("-- end --")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("filepath", help="Path to the sorted file (.npy)", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--in_out", default=1, help="1 in, -1 out of the rf", type=int)
    parser.add_argument(
        "--e_align", "-e", default=2, help="Event to aligne the spikes", type=int
    )
    parser.add_argument(
        "--cgroup", "-g", default="good", help="cluster goup, good or mua", type=str
    )
    args = parser.parse_args()
    try:
        main(args.filepath, args.output_dir, args.in_out, args.e_align, args.cgroup)
    except FileExistsError:
        logging.error("filepath does not exist")
