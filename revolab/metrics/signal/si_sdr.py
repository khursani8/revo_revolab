"""
Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric implementation.

SI-SDR is an enhancement of the standard SDR metric, designed to be invariant
to the absolute scaling of the estimated signal. This makes it a more reliable
metric for tasks where the output signal's amplitude may not be perfectly
aligned with the reference signal.
"""

import numpy as np
from typing import Union

from ..base import AudioMetric, find_test_audio_files, demo_audio_files_exist, create_demo_audio_data


class SISDR(AudioMetric):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric.

    Computes the SI-SDR between a reference and an estimated audio signal.
    This metric is particularly useful in source separation and speech enhancement
    tasks where the absolute amplitude of the estimated signal is not critical.

    A higher SI-SDR value indicates better performance.

    Typical SI-SDR values (in dB):
    - > 15 dB: Good separation/enhancement
    - 5-15 dB: Moderate quality
    - < 5 dB: Poor quality
    """

    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize SI-SDR metric.

        Args:
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

    def _compute_si_sdr(self,
                        reference: np.ndarray,
                        estimate: np.ndarray) -> float:
        """
        Compute SI-SDR between a single reference and estimate.

        Args:
            reference: Reference signal (1D array)
            estimate: Estimated signal (1D array)

        Returns:
            SI-SDR value in dB
        """
        # Ensure non-zero signals
        if np.sum(reference ** 2) < 1e-8:
            return -np.inf

        # Ensure same length
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]

        # s_target = <s', s>s / ||s||^2
        dot_product = np.dot(estimate, reference)
        norm_s_sq = np.sum(reference ** 2)
        s_target = (dot_product / norm_s_sq) * reference

        # e_noise = s' - s_target
        e_noise = estimate - s_target

        # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        s_target_energy = np.sum(s_target ** 2)
        e_noise_energy = np.sum(e_noise ** 2)

        if e_noise_energy < 1e-8:
            return np.inf

        si_sdr = 10 * np.log10(s_target_energy / e_noise_energy)

        return float(si_sdr)

    def compute(self,
                ref_audio: Union[str, np.ndarray],
                est_audio: Union[str, np.ndarray]) -> float:
        """
        Compute SI-SDR between reference and estimated audio.

        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)

        Returns:
            SI-SDR value in dB
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)

        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)

        return self._compute_si_sdr(ref_audio, est_audio)


def si_sdr(ref_audio: Union[str, np.ndarray],
           est_audio: Union[str, np.ndarray],
           **kwargs) -> float:
    """
    Convenience function to compute SI-SDR.

    Args:
        ref_audio: Reference audio (file path or array)
        est_audio: Estimated audio (file path or array)
        **kwargs: Additional parameters for SISDR

    Returns:
        SI-SDR value in dB
    """
    metric = SISDR(**kwargs)
    return metric.compute(ref_audio, est_audio)


if __name__ == "__main__":
    """
    Demo script for SI-SDR metric.
    """
    print("=== SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) Demo ===")

    si_sdr_metric = SISDR()

    if demo_audio_files_exist():
        print("\n1. Testing with real audio files...")
        audio_files = find_test_audio_files()

        if len(audio_files) >= 2:
            ref_file = audio_files[0]
            est_file = audio_files[1]

            print(f"   Reference: {ref_file}")
            print(f"   Estimated: {est_file}")

            try:
                si_sdr_value = si_sdr_metric.compute(ref_file, est_file)
                print(f"   SI-SDR: {si_sdr_value:.2f} dB")
            except Exception as e:
                print(f"   Error computing SI-SDR: {e}")
        else:
            print("   Need at least 2 audio files for SI-SDR comparison.")
    else:
        print("\n1. Testing with synthetic audio...")
        demo_data = create_demo_audio_data()
        ref_audio = demo_data['clean']

        # 1. Scaled version of the reference
        scaled_audio = 0.5 * ref_audio
        si_sdr_scaled = si_sdr_metric.compute(ref_audio, scaled_audio)
        print(f"   Clean vs Scaled (0.5x): SI-SDR = {si_sdr_scaled:.2f} dB (should be inf)")

        # 2. Noisy version
        noisy_audio = (ref_audio + 0.1 * np.random.randn(len(ref_audio))).astype(np.float32)
        si_sdr_noisy = si_sdr_metric.compute(ref_audio, noisy_audio)
        print(f"   Clean vs Noisy: SI-SDR = {si_sdr_noisy:.2f} dB")
        
        # 3. Scaled and noisy
        scaled_noisy_audio = (0.5 * ref_audio + 0.1 * np.random.randn(len(ref_audio))).astype(np.float32)
        si_sdr_scaled_noisy = si_sdr_metric.compute(ref_audio, scaled_noisy_audio)
        print(f"   Clean vs Scaled and Noisy: SI-SDR = {si_sdr_scaled_noisy:.2f} dB")

    print("\n=== SI-SDR Demo Complete ===")
