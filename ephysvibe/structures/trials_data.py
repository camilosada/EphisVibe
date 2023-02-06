import h5py
import numpy as np
from pathlib import Path
from .bhv_data import BhvData


class TrialsData(BhvData):
    def __init__(
        self,
        block: np.ndarray,
        iti: np.ndarray,
        position: np.ndarray,
        reward_plus: np.ndarray,
        trial_error: np.ndarray,
        delay_time: np.ndarray,
        fix_time: np.ndarray,
        fix_window_radius: np.ndarray,
        idletime3: np.ndarray,
        rand_delay_time: np.ndarray,
        reward_dur: np.ndarray,
        wait_for_fix: np.ndarray,
        # sacc
        sacc_code: np.ndarray,
        fix_post_sacc_blank: np.ndarray,
        max_reaction_time: np.ndarray,
        stay_time: np.ndarray,
        fix_fp_t_time: np.ndarray,
        fix_fp_post_t_time: np.ndarray,
        fix_fp_pre_t_time: np.ndarray,
        fix_close: np.ndarray,
        fix_far: np.ndarray,
        closeexc: np.ndarray,
        excentricity: np.ndarray,
        farexc: np.ndarray,
        # dmts
        eye_ml: np.ndarray,
        condition: np.ndarray,
        code_numbers: np.ndarray,
        code_times: np.ndarray,
        stim_match: np.ndarray,
        samp_pos: np.ndarray,
        stim_total: np.ndarray,
        test_distractor: np.ndarray,
        test_stimuli: np.ndarray,
        sample_time: np.ndarray,
        test_time: np.ndarray,
        # sp
        sp_samples: np.ndarray,
        eyes_values: np.ndarray,
        lfp_values: np.ndarray,
        clusters_id: np.ndarray,
        clusters_ch: np.ndarray,
        clustersgroup: np.ndarray,
        clusterdepth: np.ndarray,
        code_samples: np.ndarray,
    ):
        """Initialize the class."""
        super().__init__(
            block,
            iti,
            position,
            reward_plus,
            trial_error,
            delay_time,
            fix_time,
            fix_window_radius,
            idletime3,
            rand_delay_time,
            reward_dur,
            wait_for_fix,
            # sacc
            sacc_code,
            fix_post_sacc_blank,
            max_reaction_time,
            stay_time,
            fix_fp_t_time,
            fix_fp_post_t_time,
            fix_fp_pre_t_time,
            fix_close,
            fix_far,
            closeexc,
            excentricity,
            farexc,
            # dmts
            eye_ml,
            condition,
            code_numbers,
            code_times,
            stim_match,
            samp_pos,
            stim_total,
            test_distractor,
            test_stimuli,
            sample_time,
            test_time,
        )
        # sp
        self.sp_samples = sp_samples
        self.eyes_values = eyes_values
        self.lfp_values = lfp_values
        self.code_samples = code_samples
        self.clusters_id = clusters_id
        self.clusters_ch = clusters_ch
        self.clustersgroup = clustersgroup
        self.clusterdepth = clusterdepth

        # self._check_shapes()

    # def _check_shapes(self):
    #     n_trials, n_neurons, n_ts = self.sp_samples.shape
    #     n_codes = self.code_numbers.shape[1]
    #     if self.blocks.shape != (n_trials,):
    #         raise ValueError(
    #             "sp_samples and blocks must have the same number of trials (%d != %d)"
    #             % (n_trials, len(self.blocks))
    #         )
    #     if self.code_numbers.shape != (n_trials, n_codes):
    #         raise ValueError(
    #             "Expected shape: (%s,%s), but blocks has shape %d"
    #             % (n_trials, len(self.blocks))
    #         )
    #     if self.code_samples.shape != (n_trials, n_codes):
    #         raise ValueError(
    #             "sp_samples and blocks must have the same number of trials (%d != %d)"
    #             % (n_trials, len(self.blocks))
    #         )
    #     if self.eyes_values.shape[0] != n_trials or self.eyes_values.shape[2] != n_ts:
    #         raise ValueError(
    #             "sp_samples and blocks must have the same number of trials (%d != %d)"
    #             % (n_trials, len(self.blocks))
    #         )

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        super().to_python_hdf5(save_path)

        with h5py.File(save_path, "a") as f:
            group = f["data"]
            group.create_dataset(
                "sp_samples",
                self.sp_samples.shape,
                compression="gzip",
                data=self.sp_samples,
            )
            group.create_dataset(
                "eyes_values",
                self.eyes_values.shape,
                data=self.eyes_values,
                compression="gzip",
            )
            group.create_dataset(
                "lfp_values",
                self.lfp_values.shape,
                data=self.lfp_values,
                compression="gzip",
            )
            group.create_dataset(
                "code_samples",
                self.code_samples.shape,
                data=self.code_samples,
                compression="gzip",
            )
            group.create_dataset(
                "clusters_id",
                self.clusters_id.shape,
                data=self.clusters_id,
                compression="gzip",
            )
            group.create_dataset(
                "clusters_ch",
                self.clusters_ch.shape,
                data=self.clusters_ch,
                compression="gzip",
            )
            group.create_dataset(
                "clustersgroup",
                self.clustersgroup.shape,
                data=self.clustersgroup,
                compression="gzip",
            )
            group.create_dataset(
                "clusterdepth",
                self.clusterdepth.shape,
                data=self.clusterdepth,
                compression="gzip",
            )
        f.close()

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        # load the data

        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["data"]
            sp_samples = group["sp_samples"][:]
            code_samples = group["code_samples"][:]
            eyes_values = group["eyes_values"][:]
            lfp_values = group["lfp_values"][:]

            clusters_id = group["clusters_id"][:]
            clusters_ch = group["clusters_ch"][:]
            clustersgroup = np.array(group["clustersgroup"], dtype=str)
            clusterdepth = group["clusterdepth"][:]
        # create class object and return
        bhv_dict = super().from_python_hdf5(load_path)
        trials_data = {
            **bhv_dict,
            "sp_samples": sp_samples,
            "code_samples": code_samples,
            "eyes_values": eyes_values,
            "lfp_values": lfp_values,
            "clusters_id": clusters_id,
            "clusters_ch": clusters_ch,
            "clustersgroup": clustersgroup,
            "clusterdepth": clusterdepth,
        }
        return cls(**trials_data)
