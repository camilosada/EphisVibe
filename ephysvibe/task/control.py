import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def correct_task_errors(
    trial_errors: np.ndarray,
    stim_match: np.ndarray,
    test_stimuli: np.ndarray,
    sample_ids: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Corrects trial errors based on stimulus matching conditions.

    The function performs the following:
      1. Identifies trials where a match is in an unexpected position and marks these
          with an error code (-3).
      2. For trials flagged with error -3:
            - If the original error was break fixation (code 3), it is left unchanged.
            - If the original error was correct (code 0), indicating a miss, it is changed to 8.
            - If the error was due to a bar release (code 6), the last valid test stimulus is
              checked. If it matches the sample ID, the error is corrected to 0; otherwise, it
              remains 6.
    Args:
        trial_errors (np.ndarray): trial error codes.
        stim_match (np.ndarray): expected stimulus match positions.
        test_stimuli (np.ndarray): test stimuli values.
        sample_ids (np.ndarray): sample IDs.
        verbose (bool, optional): If True, logs messages about the error correction process. Defaults to True.

    Returns:
        np.ndarray: corrected errors.
    """
    task_errors = trial_errors.copy()
    # --- Step 1: Identify Unexpected Matches ---
    match_mask = test_stimuli == sample_ids[:, np.newaxis]
    match_rows, match_cols = np.where(match_mask)
    unexpected_match_idx = np.where(stim_match[match_rows] - 3 != match_cols)[0]
    if unexpected_match_idx.size == 0:
        if verbose:
            logger.info("No errors found.")
        return task_errors  # Nothing to correct
    if unexpected_match_idx.size > 0:  # Mark these trials with error code -3
        task_errors[match_rows[unexpected_match_idx]] = -3
    # --- Step 2: Replace Wrong Errors with Correct Ones ---
    error_indices = np.where(task_errors == -3)[0]
    corrected_errors = task_errors[error_indices].copy()
    original_errors = trial_errors[error_indices]
    # Replace with correct error codes
    corrected_errors[original_errors == 3] = 3
    corrected_errors[original_errors == 0] = 8
    bar_release_mask = original_errors == 6
    bar_release_indices = np.where(bar_release_mask)[0]
    # check if it was released correctly
    if bar_release_indices.size > 0:
        test_stimuli_subset = test_stimuli[error_indices]
        no_nan_tests = ~np.isnan(test_stimuli_subset)
        last_valid_idx = (no_nan_tests.shape[1] - 1) - np.argmax(
            no_nan_tests[:, ::-1], axis=1
        )
        last_valid_idx[~no_nan_tests.any(axis=1)] = -1
        last_test_values = test_stimuli_subset[
            bar_release_indices, last_valid_idx[bar_release_indices]
        ]
        corresponding_sample_ids = sample_ids[error_indices][bar_release_indices]
        match_check = last_test_values == corresponding_sample_ids
        corrected_errors[bar_release_indices[match_check]] = 0
        corrected_errors[bar_release_indices[~match_check]] = 6
    # Finally, update the trial errors for the processed error trials
    task_errors[error_indices] = corrected_errors
    if verbose:
        logger.info(f"{unexpected_match_idx.size} errors found.")
    return task_errors
