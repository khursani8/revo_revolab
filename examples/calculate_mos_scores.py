import os
import torch
import glob
import csv
import librosa
import numpy as np
from tqdm import tqdm
# Note: sidon_recon is a private/local dependency that must be provided by the user.
from sidon_recon import denoise_speech, fe_model, decoder_model

def calculate_mos(predictor, wav, sr):
    try:
        # Normalize audio
        if np.max(np.abs(wav)) > 1.0:
            wav = wav / np.max(np.abs(wav))

        # Convert to torch tensor
        wave_tensor = torch.from_numpy(wav).unsqueeze(0)

        # Predict the MOS score
        score = predictor(wave_tensor, sr)
        
        return score.item()
    except Exception as e:
        # print(f"Error calculating MOS: {e}") # Removed for tqdm
        return None

def main():
    # Download the model once to cache it
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    # Directories to scan
    dirs_to_scan = ['shrakat_uploaded_1k', 'Malay_sample']
    
    # Target sample rates
    target_sample_rates = [8000, 16000, 22000]

    output_csv_file = 'mos_scores_all_with_recon.csv'
    
    # Read processed files
    processed_files = set()
    if os.path.exists(output_csv_file):
        with open(output_csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            try:
                next(reader) # skip header
                for row in reader:
                    processed_files.add(row[0])
            except StopIteration:
                pass # empty file

    all_files = []
    for directory in dirs_to_scan:
        all_files.extend(glob.glob(os.path.join(directory, '*.mp3')))
        
    files_to_process = [f for f in all_files if f not in processed_files]

    # Write results to CSV
    with open(output_csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not processed_files:
            writer.writerow(['filename', 'original_sr', 'mos_original', 'mos_8k', 'mos_16k', 'mos_22k', 'mos_recon_48k', 'mos_recon_8k', 'mos_recon_16k', 'mos_recon_22k'])
        
        # Wrap the loop with tqdm for a progress bar
        for file_path in tqdm(files_to_process, desc="Processing audio files"):
            try:
                wav_orig, sr_orig = librosa.load(file_path, sr=None)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            mos_scores = {'filename': file_path, 'original_sr': sr_orig}

            # Calculate MOS for original
            mos_original = calculate_mos(predictor, wav_orig, sr_orig)
            if mos_original is None:
                continue
            mos_scores['mos_original'] = mos_original

            for target_sr in target_sample_rates:
                if sr_orig == target_sr:
                    mos_scores[f'mos_{target_sr // 1000}k'] = mos_original
                else:
                    try:
                        wav_resampled = librosa.resample(wav_orig, orig_sr=sr_orig, target_sr=target_sr)
                        mos_resampled = calculate_mos(predictor, wav_resampled, target_sr)
                        mos_scores[f'mos_{target_sr // 1000}k'] = mos_resampled
                    except Exception as e:
                        print(f"Error resampling or calculating MOS for {file_path} at {target_sr}Hz: {e}")
                        mos_scores[f'mos_{target_sr // 1000}k'] = None
            
            # sidon recon part
            try:
                sr_recon, wav_recon_int = denoise_speech(file_path, fe_model, decoder_model)

                if wav_recon_int is not None:
                    # convert to float and flatten
                    wav_recon = (wav_recon_int.astype(np.float32) / 32767.0).flatten()

                    # MOS for reconstructed
                    mos_scores['mos_recon_48k'] = calculate_mos(predictor, wav_recon, sr_recon)

                    # MOS for resampled reconstructed
                    for target_sr in target_sample_rates:
                        try:
                            wav_recon_resampled = librosa.resample(wav_recon, orig_sr=sr_recon, target_sr=target_sr)
                            mos_scores[f'mos_recon_{target_sr // 1000}k'] = calculate_mos(predictor, wav_recon_resampled, target_sr)
                        except Exception as e:
                            print(f"Error resampling recon {file_path} to {target_sr}: {e}")
                            mos_scores[f'mos_recon_{target_sr // 1000}k'] = None
                else:
                    mos_scores['mos_recon_48k'] = None
                    for target_sr in target_sample_rates:
                         mos_scores[f'mos_recon_{target_sr // 1000}k'] = None
            
            except Exception as e:
                print(f"Error during sidon recon for {file_path}: {e}")
                mos_scores['mos_recon_48k'] = None
                for target_sr in target_sample_rates:
                    mos_scores[f'mos_recon_{target_sr // 1000}k'] = None


            writer.writerow([mos_scores.get('filename'), mos_scores.get('original_sr'), mos_scores.get('mos_original'), 
                             mos_scores.get('mos_8k'), mos_scores.get('mos_16k'), mos_scores.get('mos_22k'),
                             mos_scores.get('mos_recon_48k'), mos_scores.get('mos_recon_8k'),
                             mos_scores.get('mos_recon_16k'), mos_scores.get('mos_recon_22k')])

if __name__ == '__main__':
    main()
