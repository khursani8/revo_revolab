> **Unique:** Python speech-evaluation metrics library: MCD, F0 correlation/RMSE, SDR family, PESQ/STOI, and MOS predictors (UTMOS via torch.hub) for scoring TTS/ASR output quality.

# Revolab: Comprehensive Speech Evaluation Library

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UV](https://img.shields.io/badge/UV-Compatible-purple.svg)](https://github.com/astral-sh/uv)

Revolab is a comprehensive Python library for speech evaluation metrics, providing a wide range of tools to assess speech quality, intelligibility, and similarity. It's designed for researchers, engineers, and developers working with speech processing, synthesis, and enhancement systems.

## 🚀 Features

- **Comprehensive Metrics**: F0/Mel-cepstral, Signal Quality, Perceptual Quality, Discrete Speech, and Subjective metrics
- **Easy-to-Use API**: Clean, intuitive imports and usage patterns
- **UV Compatible**: Modern Python package management with `uv sync`
- **Interactive Demos**: Every metric includes a `__main__` section for quick testing
- **Flexible Input**: Works with file paths or numpy arrays
- **Extensible Architecture**: Modular design for easy extension

## 📦 Installation

### With UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd revolab

# Install dependencies
uv sync

# Run demos
uv run python -m revolab.metrics.f0_mcd.mcd
uv run python -m revolab.metrics.signal.sdr
```

### With pip (Coming Soon)

```bash
pip install revolab
```

## 🎯 Quick Start

### Import and Use Metrics

```python
# Import specific metrics (recommended)
from revolab.metrics import mcd, f0_corr, f0_rmse, sdr, si_sdr, sir, sar, pesq, stoi

# Import by category
from revolab.metrics.f0_mcd import mcd, f0_corr, f0_rmse
from revolab.metrics.signal import sdr, si_sdr, sir, sar
from revolab.metrics.quality import pesq, stoi

# Use with file paths
mcd_score = mcd('reference.wav', 'estimated.wav')
sdr_score = sdr('clean.wav', 'enhanced.wav')
pesq_score = pesq('clean.wav', 'enhanced.wav')
stoi_score = stoi('clean.wav', 'enhanced.wav')


# Use with audio arrays
import librosa
ref_audio = librosa.load('reference.wav', sr=16000)[0]
est_audio = librosa.load('estimated.wav', sr=16000)[0]
f0_score = f0_corr(ref_audio, est_audio)
rmse_score = f0_rmse(ref_audio, est_audio)
si_sdr_score = si_sdr(ref_audio, est_audio)

print(f"MCD: {mcd_score:.3f} dB")
print(f"F0 Correlation: {f0_score:.3f}")
print(f"F0 RMSE: {rmse_score:.2f} Hz")
print(f"SDR: {sdr_score:.2f} dB")
print(f"SI-SDR: {si_sdr_score:.2f} dB")
print(f"PESQ: {pesq_score:.2f}")
print(f"STOI: {stoi_score:.4f}")
```

### Run Interactive Demos

Each metric includes an interactive demo for quick testing:

```bash
# Test individual metrics
uv run python -m revolab.metrics.f0_mcd.mcd
uv run python -m revolab.metrics.f0_mcd.f0_corr
uv run python -m revolab.metrics.f0_mcd.f0_rmse
uv run python -m revolab.metrics.signal.sdr
uv run python -m revolab.metrics.signal.si_sdr
uv run python -m revolab.metrics.signal.sir
uv run python -m revolab.metrics.signal.sar
uv run python -m revolab.metrics.signal.si_snri
uv run python -m revolab.metrics.quality.pesq
uv run python -m revolab.metrics.quality.stoi
```

## 📊 Available Metrics

### F0 and Mel-Cepstral Metrics

Evaluate fundamental frequency and spectral characteristics:

- **`mcd`**: Mel Cepstral Distortion - Spectral distance in mel-cepstral domain
- **`f0_corr`**: F0 Correlation - Pearson correlation of F0 contours
- **`f0_rmse`**: F0 Root Mean Square Error - Average F0 deviation in Hz

```python
from revolab.metrics import mcd, f0_corr, f0_rmse

# Evaluate spectral quality
mcd_score = mcd(reference, estimated)

# Evaluate pitch tracking
f0_correlation = f0_corr(reference, estimated)
f0_error = f0_rmse(reference, estimated)
```

### Signal Quality Metrics

Assess overall signal quality and separation performance:

- **`sdr`**: Signal to Distortion Ratio - Ratio of desired signal to total distortion
- **`si_sdr`**: Scale-Invariant Signal to Distortion Ratio
- **`sir`**: Signal to Interference Ratio
- **`sar`**: Signal to Artifact Ratio
- **`si_snri`**: Scale-Invariant Signal to Noise Ratio Improvement

```python
from revolab.metrics import sdr, si_sdr, si_snri

# Evaluate signal enhancement
sdr_score = sdr(clean_reference, enhanced_output)
si_sdr_score = si_sdr(clean_reference, enhanced_output)
si_snri_score = si_snri(clean_reference, noisy_input, enhanced_output)
```

### Quality Metrics

Perceptual quality assessment:

- **`pesq`**: Perceptual Evaluation of Speech Quality
- **`stoi`**: Short-Time Objective Intelligibility
- **`visqol`**: Virtual Speech Quality Objective Listener (coming soon)

```python
from revolab.metrics import pesq, stoi

# Evaluate perceptual quality
pesq_score = pesq(clean_reference, enhanced_output)
stoi_score = stoi(clean_reference, enhanced_output)
```


### Discrete Speech Metrics (Coming Soon)

Token-level speech evaluation:

- **`speech_bert`**: Speech BERT Score
- **`speech_bleu`**: Speech BLEU Score
- **`speech_token_distance`**: Speech Token Distance Score

### Subjective Metrics (Coming Soon)

Pseudo-subjective quality scores:

- **`utmos`**: UT-MOS Score
- **`dnsmos`**: DNS-MOS Score
- **`plcmos`**: PLC-MOS Score
- **`aecmos`**: AEC-MOS Score

## 🧪 Testing with Audio Files

1. Add your test audio files to the `test_data/` directory
2. Follow the format requirements in `test_data/README.md`
3. Run demos to test with your files:

```bash
# Add audio files first
cp your_audio_files/*.wav test_data/

# Test with real audio
uv run python -m revolab.metrics.f0_mcd.mcd
```

## 🏗️ Architecture

```
revolab/
├── metrics/
│   ├── f0_mcd/          # F0 and Mel-Cepstral metrics
│   │   ├── mcd.py       # Mel Cepstral Distortion
│   │   ├── f0_corr.py   # F0 Correlation
│   │   └── f0_rmse.py   # F0 RMSE
│   ├── signal/           # Signal quality metrics
│   │   ├── sdr.py       # Signal to Distortion Ratio
│   │   ├── si_sdr.py    # Scale-Invariant SDR
│   │   ├── sir.py       # Signal to Interference Ratio
│   │   ├── sar.py       # Signal to Artifact Ratio
│   │   └── si_snri.py   # SI-SNR Improvement
│   ├── quality/          # Perceptual quality metrics
│   │   ├── pesq.py      # Perceptual Evaluation of Speech Quality
│   │   └── stoi.py      # Short-Time Objective Intelligibility
│   ├── discrete/         # Discrete speech metrics
│   ├── subjective/       # Subjective metrics
│   └── base.py          # Base classes and utilities
├── models/              # Speech models (future)
├── data/               # Data processing utilities (future)
├── utils/              # General utilities
└── test_data/          # Your test audio files
```

## 🔧 Configuration

All metrics support customizable parameters:

```python
from revolab.metrics.f0_mcd import MCD

# Custom MCD configuration
mcd_metric = MCD(
    n_mfcc=13,           # Number of MFCC coefficients
    n_fft=1024,          # FFT window size
    hop_length=256,       # Hop length
    n_mels=80,           # Number of mel filters
    sample_rate=16000      # Sample rate
)

# Use custom configuration
score = mcd_metric(reference, estimated)
```

## 📈 Quality Benchmarks

### MCD (Mel Cepstral Distortion)
- **Excellent**: < 1.0 dB
- **Good**: 1.0-2.5 dB
- **Fair**: 2.5-4.0 dB
- **Poor**: > 4.0 dB

### F0 Correlation
- **Excellent**: 0.8-1.0
- **Good**: 0.6-0.8
- **Fair**: 0.4-0.6
- **Poor**: 0.0-0.4

### F0 RMSE
- **Excellent**: < 10 Hz
- **Good**: 10-20 Hz
- **Fair**: 20-40 Hz
- **Poor**: > 40 Hz

### SDR (Signal to Distortion Ratio)
- **Excellent**: > 20 dB
- **Good**: 10-20 dB
- **Fair**: 0-10 dB
- **Poor**: < 0 dB

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement your metric with proper tests and demos
4. Submit a pull request

### Adding New Metrics

1. Create a new file in the appropriate category directory
2. Inherit from `AudioMetric` or `BaseMetric`
3. Implement required methods: `compute()`
4. Add convenience function and `__main__` demo
5. Update `__init__.py` files
6. Add tests

```python
from ..base import AudioMetric

class YourMetric(AudioMetric):
    def compute(self, ref_audio, est_audio):
        # Your implementation
        return score

def your_metric(ref_audio, est_audio, **kwargs):
    metric = YourMetric(**kwargs)
    return metric.compute(ref_audio, est_audio)

if __name__ == "__main__":
    # Demo code
    pass
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [librosa](https://librosa.org/) for audio processing
- Inspired by various speech evaluation research
-感谢 the speech processing community

## 📞 Contact

- **Email**: contact@revolab.ai
- **Issues**: [GitHub Issues](https://github.com/your-org/revolab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/revolab/discussions)

---

**Revolab** - Empowering Speech Evaluation 🎤✨
