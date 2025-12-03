# Test Data for Revolab

This folder contains test audio files for demonstrating and testing the revolab speech metrics library.

## Expected Audio Format

For consistent testing across all metrics, please ensure your audio files follow these specifications:

- **Format**: WAV (recommended)
- **Sample Rate**: 16kHz (most speech metrics work best with this)
- **Bit Depth**: 16-bit
- **Channels**: Mono (single channel)
- **Duration**: 1-10 seconds per file

## File Naming Convention

Please name your files as:
- `sample1.wav`
- `sample2.wav`
- `sample3.wav`
- ...
- `sample10.wav`

## Usage

Once you add your audio files here, you can run demos like:

```bash
# Test individual metrics
uv run python -m revolab.metrics.f0_mcd.mcd
uv run python -m revolab.metrics.signal.sdr
uv run python -m revolab.metrics.quality.pesq
```

The demo scripts will automatically detect and use the audio files in this directory.

## Notes

- Make sure audio files are not too large (keep them under 10MB each)
- Clean speech without background noise works best for most metrics
- Some metrics may need paired files (e.g., reference and enhanced versions)
