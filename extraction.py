import os
import librosa
import numpy as np
import soundfile as sf
import csv

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = sf.read(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def extract_jitter_shimmer(file_path):
    y, sr = sf.read(file_path)
    rms = np.sqrt(np.mean(y**2))
    jitter = 20 * np.log10(rms)
    shimmer = np.abs(20 * np.log10(np.max(np.abs(y)))) - jitter
    return jitter, shimmer

def write_to_csv(file_path, mfccs, jitter, shimmer, file_name, gender, health):
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for i in range(mfccs.shape[1]):
            row = [file_name, gender, health] + mfccs[:, i].tolist() + [jitter, shimmer]
            csv_writer.writerow(row)

def process_audio_files(directory, output_csv_path):
    # Initialize CSV file with header
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['File_name', 'GENDER', 'HEALTH'] + ['MFCC_{}'.format(i+1) for i in range(13)] + ['Jitter', 'Shimmer']
        csv_writer.writerow(header)

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            
            # Extract MFCC
            mfccs = extract_mfcc(file_path)
            print(f"MFCCs for {filename}:")
            print(mfccs)
            
            # Extract jitter and shimmer
            jitter, shimmer = extract_jitter_shimmer(file_path)
            print(f"Jitter for {filename}:", jitter)
            print(f"Shimmer for {filename}:", shimmer)
            
            # Write to CSV, including file name, gender, and health status
            write_to_csv(output_csv_path, mfccs, jitter, shimmer, filename, "Female", "Unhealthy")
    
    print(f"Data for all files written to {output_csv_path}")


directory_path = 'a/Unhealthy/Female' 
output_csv_path = 'featuresFemaleUnHel.csv'

process_audio_files(directory_path, output_csv_path)
