import numpy as np


def compute_performance(
    test_stimuli: np.ndarray, sample_id: np.ndarray, trial_error: np.ndarray
) -> np.ndarray:
    
    """Computes a performance matrix based on test stimuli, sample IDs, and task error codes.

        The performance matrix matches test_stimuli matrix and assigns performance codes as follows:
        10: Hit
        11: Correct Rejection
        12: Miss
        13: False Alarm
        14: Break Fixation

        Unassigned positions remain as np.nan.
        Args:
            test_stimuli (np.ndarray): array (trials x tests) with test stimulus values.
            sample_id (np.ndarray): array of sample IDs for each trial.
            trial_error (np.ndarray): array with error codes for each trial.
                        Expected error codes include:
                            0: Correct
                            8: Miss
                            6: False Alarm
        Returns:
            np.ndarray: array with the same shape as test_stimuli containing
                the performance codes assigned to each test stimulus.
    """
    perf = np.full(test_stimuli.shape, np.nan)
    # HITS:
    # A hit is determined for correct (error code 0) and non-neutral (sample_id != 0) trials.
    mask_error0 = trial_error == 0
    mask_non_neutral = sample_id != 0
    mask_correct_non_neutral = np.logical_and(mask_error0, mask_non_neutral)
    match_mask = test_stimuli == sample_id[:, np.newaxis]
    # For correct non-neutral trials, assign code 1 where a match is present.
    perf[mask_correct_non_neutral] = np.where(
        match_mask[mask_correct_non_neutral], 10, perf[mask_correct_non_neutral]
    )
    # CR:
    # 1. All test positions prior to a hit are considered CR.
    mask_cr = np.full(perf.shape, False)
    hit_rows, hit_cols = np.where(perf == 1)
    for r, c in zip(hit_rows, hit_cols):
        mask_cr[r, :c] = True
    perf[mask_cr] = 11
    # 2. Catch trials are defined as trials where the maximum number of tests were presented,
    #    the trial is correct & non-neutral, and no match occurred in the last test.
    mask_max_test = np.all(~np.isnan(test_stimuli), axis=1)
    mask_catch_trials = np.logical_and(mask_max_test, mask_correct_non_neutral)
    mask_catch_trials = np.logical_and(mask_catch_trials, ~match_mask[:, -1])
    perf[mask_catch_trials] = 11
    # MISS:
    # For trials with task error 8, assign a miss code (3) at positions where a match occurred.
    mask_error8 = trial_error == 8
    if np.any(mask_error8):
        # For the subset of trials with error 8, compute a match between test stimuli and sample_id.
        sub_test_stimuli = test_stimuli[mask_error8]
        sub_sample_id = sample_id[mask_error8]
        row_match, col_match = np.where(
            sub_test_stimuli == sub_sample_id[:, np.newaxis]
        )
        perf[mask_error8, col_match] = 12
    # FALSE ALARM:
    # For trials with task error 6, mark the performance code (4) at the last valid test stimulus.
    mask_error6 = trial_error == 6
    if np.any(mask_error6):
        valid_mask = ~np.isnan(test_stimuli)
        last_valid_pos = (valid_mask.shape[1] - 1) - np.argmax(
            valid_mask[:, ::-1], axis=1
        )
        perf[mask_error6, last_valid_pos[mask_error6]] = 13

    mask_error3 = trial_error == 3
    if np.any(mask_error3):
        valid_mask = ~np.isnan(test_stimuli)
        last_valid_pos = (valid_mask.shape[1] - 1) - np.argmax(
            valid_mask[:, ::-1], axis=1
        )
        idx=np.where(~np.any(valid_mask,axis=1))[0]
        mask_error3[idx]=False
        perf[mask_error3, last_valid_pos[mask_error3]] = 14
    return perf