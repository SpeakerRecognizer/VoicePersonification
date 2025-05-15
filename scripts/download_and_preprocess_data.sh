#!/bin/bash

# This script downloades and prepares all the necessary data for running the VoicePersonification pipeline in 5 stages.
# stage 1 for downloading all the data in archives 
# stage 2 for decompressing the archives
# stage 3 for converting data to wav format
# stage 4 for creating wav.scp and utt2spk files
# stage 5 for converting VoxCeleb1-O (Cleaned) trials in kaldi format
# Initialize stage and stop_stage depending on the stage you want to start from and the one you want to stop on.
# By default the script runs all the stages one by one: stage=1, stop_stage=5.

# The script requires 4 arguments: 
# 1. download_dir --- desired path to downloaded files (archives and trials)
# 2. rawdata_dir --- desired path to decompressed archives 
# 3. scp_dir --- desired path to scp files
# 4. protocols_dir --- desired path to kaldi protocols.
# Example of usage: 
# $ bash scripts/download_and_preprocess_data.sh data/download_data data/raw_data data/scp data/protocols


stage=1
stop_stage=5


download_dir=$1
rawdata_dir=$2
scp_dir=$3
protocols_dir=$4


# download VoxCeleb1-O (Cleaned) trials and archives containing voxceleb1_test, voxceleb1_dev, voxceleb2_dev
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download vox1_test_wav.zip, vox1_dev_wav.zip, vox2_aac.zip and VoxCeleb1-O (Cleaned) trials"
  echo "This may take a long time..."
  echo "As an alternative, you can download all the archives aside from this script by calling $ bash scripts/data_scripts/download_data.sh <download_path>"

  bash scripts/data_scripts/download_data.sh ${download_dir}
  echo "Data downloaded into $download_dir!"
fi


# decompress downloaded archives: vox1_test_wav.zip, vox1_dev_wav.zip, vox2_acc.zip 
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."
  echo "As an alternative, you can decompress all the archives aside from this script by calling $ bash scripts/data_scripts/decompress_archives.sh <download_path> <raw_data_path>"

  bash scripts/data_scripts/decompress_archives.sh ${download_dir} ${rawdata_dir}
  echo "Decompressed audios can be found in $rawdata_dir!"
fi


# convert voxceleb2 audios from m4a to wav format
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Convert voxceleb2 audios from m4a to wav using ffmpeg."
  echo "This could also take some time ..."
  echo "As an alternative, you can convert audios aside from this script by calling $ bash scripts/data_scripts/convert_to_wav.sh <voxceleb2_m4a_path> <voxceleb2_wav_path>"
  
  bash scripts/data_scripts/convert_to_wav.sh ${rawdata_dir}/voxceleb2_m4a ${rawdata_dir}/voxceleb2_wav
  echo "Converted audios can be found in ${rawdata_dir}/voxceleb2_wav"
fi


# prepare wav.scp and utt2spk files for voxceleb1_dev voxceleb1_test and voxceleb2
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for each dataset..."
  echo "This could also take some time..."
  echo "As an alternative, you can create these files aside from this script for each dataset by calling $ python scripts/data_scripts/create_scp_files.py <wav_path> <scp_path>"
  
  #voxceleb1_test
  echo "Prepare scp files for voxceleb1_test..."
  python scripts/data_scripts/create_scp_files.py --wav_path ${rawdata_dir}/voxceleb1/test/wav --scp_path ${scp_dir}/voxceleb1/test
  
  #voxceleb1_dev
  echo "Prepare scp files for voxceleb1_dev..."
  python scripts/data_scripts/create_scp_files.py --wav_path ${rawdata_dir}/voxceleb1/dev/wav --scp_path ${scp_dir}/voxceleb1/dev

  #voxceleb2_dev
  echo "Prepare scp files for voxceleb2_dev..."
  python scripts/data_scripts/create_scp_files.py --wav_path ${rawdata_dir}/voxceleb2_wav/ --scp_path ${scp_dir}/voxceleb2/dev

  echo "Success! Check scp files in $scp_dir!"
fi

# prepare protocols for VoxCeleb1-O (Cleaned) in kaldi format
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Prepare protocols in kaldi format..."
  echo "As an alternative, you can create protocols aside from this script by calling $ python scripts/data_scripts/create_protocols.py <trials_path> <protocols_path>"

  python scripts/data_scripts/create_protocols.py --trials_path ${download_dir}/trials/vox1-O\(cleaned\).txt --protocols_path $protocols_dir
  echo "Success! Check $protocols_dir!"
fi