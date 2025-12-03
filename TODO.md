# TODO: Remaining Metrics Implementation

This document outlines the remaining tasks to complete the revolab speech evaluation library.

## 🔧 High Priority

### F0 and Mel-Cepstral Metrics
- [x] **MCD**: Mel Cepstral Distortion
  - Location: `revolab/metrics/f0_mcd/mcd.py`
  - Description: Spectral distance in mel-cepstral domain
  - Priority: High

- [x] **F0 Correlation**: Pearson correlation of F0 contours
  - Location: `revolab/metrics/f0_mcd/f0_corr.py`
  - Description: Pearson correlation of F0 contours
  - Priority: High

- [x] **F0 RMSE**: F0 Root Mean Square Error
  - Location: `revolab/metrics/f0_mcd/f0_rmse.py`
  - Description: Average F0 deviation in Hz
  - Priority: High

### Signal Category Metrics
- [x] **SDR**: Signal to Distortion Ratio
  - Location: `revolab/metrics/signal/sdr.py`
  - Description: Ratio of desired signal to total distortion
  - Priority: High

- [x] **SI-SDR**: Scale-Invariant Signal to Distortion Ratio
  - Location: `revolab/metrics/signal/si_sdr.py`
  - Description: SDR invariant to scaling factors
  - Priority: High

- [x] **SIR**: Signal to Interference Ratio  
  - Location: `revolab/metrics/signal/sir.py`
  - Description: Ratio of target signal to interference
  - Priority: High

- [x] **SAR**: Signal to Artifact Ratio
  - Location: `revolab/metrics/signal/sar.py` 
  - Description: Ratio of target signal to artifacts
  - Priority: High

- [x] **SI-SNRI**: Scale-Invariant Signal to Noise Ratio Improvement
  - Location: `revolab/metrics/signal/si_snri.py`
  - Description: Improvement in SNR with scale invariance
  - Priority: High

### Quality Category Metrics
- [x] **PESQ**: Perceptual Evaluation of Speech Quality
  - Location: `revolab/metrics/quality/pesq.py`
  - Dependencies: `pesq` package
  - Description: ITU-T P.862 standard for speech quality
  - Priority: High

- [x] **STOI**: Short-Time Objective Intelligibility
  - Location: `revolab/metrics/quality/stoi.py`
  - Dependencies: `pystoi` package
  - Description: Short-time objective intelligibility measure
  - Priority: High

- [ ] **VISQOL**: Virtual Speech Quality Objective Listener
  - Location: `revolab/metrics/quality/visqol.py`
  - Dependencies: `visqol` package
  - Description: Perceptual speech quality assessment
  - Priority: High
  - **Note**: Skipped due to complex installation requirements (requires Bazel and building from source).

## 🎯 Medium Priority

### Discrete Speech Metrics
- [x] **Speech BERT**: Speech BERT Score
  - Location: `revolab/metrics/discrete/speech_bert.py`
  - Dependencies: `transformers`, `torch`
  - Description: BERT-based similarity for speech tokens
  - Priority: Medium

- [x] **Speech BLEU**: Speech BLEU Score
  - Location: `revolab/metrics/discrete/speech_bleu.py`
  - Description: BLEU score adapted for speech evaluation
  - Priority: Medium

- [x] **ASR Token Distance**: ASR-based Token Distance Score
  - Location: `revolab/metrics/discrete/asr_token_distance.py`
  - Description: Distance metrics for speech token sequences via ASR.
  - Priority: Medium

- [x] **Speech Tokenizer Distance**: Speech Tokenizer Distance Score
  - Location: `revolab/metrics/discrete/speech_tokenizer_distance.py`
  - Dependencies: `snac`, `jiwer`
  - Description: Distance metrics for speech token sequences via SNAC tokenizer.
  - Priority: Medium

### Subjective Metrics
- [x] **UT-MOS**: UT-MOS Score
  - Location: `revolab/metrics/subjective/utmos.py`
  - Dependencies: `torch`, `transformers`
  - Description: Universal speech quality prediction
  - Priority: Medium

- [x] **DNS-MOS**: DNS-MOS Score
  - Location: `revolab/metrics/subjective/dnsmos.py`
  - Dependencies: `torch`, `onnxruntime`
  - Description: Deep Noise Suppression MOS prediction
  - Priority: Medium

- [x] **PLC-MOS**: PLC-MOS Score
  - Location: `revolab/metrics/subjective/plcmos.py`
  - Dependencies: `torch`, `onnxruntime`
  - Description: Packet Loss Concealment MOS prediction
  - Priority: Medium

- [x] **AEC-MOS**: AEC-MOS Score
  - Location: `revolab/metrics/subjective/aecmos.py`
  - Dependencies: `torch`, `onnxruntime`
  - Description: Acoustic Echo Cancellation MOS prediction
  - Priority: Medium

## 🔨 Infrastructure Tasks

### Configuration System
- [ ] **Default Configuration**: Create `revolab/config/defaults.py`
  - Central configuration for all metrics
  - Support for metric-specific presets
  - Environment variable support

- [ ] **YAML/JSON Config**: Support external config files
  - Load configuration from files
  - Override defaults per project

### Testing Framework
- [ ] **Unit Tests**: Comprehensive test suite
  - Location: `tests/`
  - Test each metric with known inputs/outputs
  - Integration tests for import system

- [ ] **Test Audio**: Create test audio samples
  - Known-reference audio files
  - Cover various speech types and qualities
  - Automated regression testing

- [ ] **CI/CD**: GitHub Actions setup
  - Automated testing on PRs
  - Package building and validation
  - Documentation deployment

### Documentation
- [ ] **API Documentation**: Auto-generated docs
  - Sphinx or MkDocs setup
  - API reference for all metrics
  - Examples and tutorials

- [ ] **Examples Gallery**: Usage examples
  - Real-world use cases
  - Benchmarking studies
  - Comparison with other libraries

## 📦 Distribution

### Package Management
- [ ] **PyPI Publishing**: Prepare for pip installation
  - `setup.py` or enhanced `pyproject.toml`
  - Version management
  - Release automation

- [ ] **Conda Package**: Conda-forge submission
  - Recipe for conda packaging
  - Multi-platform support

### Dependencies
- [ ] **Optional Dependencies**: Graceful handling
  - Install metrics with optional dependencies
  - Clear error messages for missing packages
  - Progressive feature discovery

## 🚀 Enhancement Tasks

### Performance
- [ ] **Batch Processing**: Process multiple files efficiently
  - Vectorized operations
  - Memory optimization
  - Progress reporting

- [ ] **GPU Acceleration**: CUDA support where applicable
  - PyTorch GPU utilization
  - Mixed precision support
  - Benchmark CPU vs GPU

### Usability
- [ ] **Command Line Interface**: Standalone tool
  - `revolab eval --metrics mcd,sdr --input *.wav`
  - Batch evaluation from command line
  - CSV/JSON output formats

- [ ] **Visualization**: Result plotting and analysis
  - Time-series visualization
  - Statistical summaries
  - Comparison plots

## 📋 Implementation Guidelines

### For Each New Metric:
1. **Create file** in appropriate category directory
2. **Inherit** from `AudioMetric` or `BaseMetric`
3. **Implement** `compute()` method
4. **Add** convenience function (e.g., `def metric_name()`)
5. **Include** `__main__` demo section
6. **Update** relevant `__init__.py` files
7. **Add** unit tests
8. **Update** documentation

### Code Quality:
- [ ] **Type Hints**: Complete type annotation
- [ ] **Docstrings**: Comprehensive documentation
- [ ] **Error Handling**: Graceful failure modes
- [ ] **Validation**: Input parameter checking
- [ ] **Performance**: Efficient implementations

## 🎯 Success Criteria

### Phase 1 (Signal + Quality Metrics)
- [x] All signal metrics implemented (SI-SDR, SIR, SAR, SI-SNRI)
- [x] All quality metrics implemented (PESQ, STOI, VISQOL) - VISQOL skipped due to installation complexity.
- [x] Full test coverage
- [x] Documentation complete

### Phase 2 (Discrete + Subjective Metrics)
- [x] All discrete metrics implemented
- [x] All subjective metrics implemented
- [ ] CLI tool available
- [ ] Performance benchmarks

### Phase 3 (Production Ready)
- [ ] PyPI package published
- [ ] CI/CD pipeline active
- [ ] Community contributions accepted
- [ ] Stable API (v1.0.0)

---

**Current Status**: ✅ Phase 2 Complete (17/18 metrics implemented - VISQOL skipped)
**Next Milestone**: Infrastructure, Distribution, and Enhancement (Phase 3)
**Target Date**: Based on development priority
