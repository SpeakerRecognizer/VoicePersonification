#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
  echo "ffmpeg could not be found. Please install it."
  exit 1
fi

# Check if a directory path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

directory="$1"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory '$directory' not found."
  exit 1
fi

# Check if a directory path is provided as an argument
if [ -z "$2" ]; then
  echo "Usage: $0 <directory_path> <destination_path>"
  exit 1
fi

destination="$2"

# Check if the directory exists
if [ ! -d "$destination" ]; then
  mkdir $destination
fi

# Find all video files in the directory and its subdirectories
find "$directory" -type f \( -name "*.m4a" \) -print0 |

# Use xargs to process the video files in parallel
xargs -0 -n 1 -P 16 bash -c '
  base_name=$(realpath -s --relative-to="$0" "${2%.*}")
  audio_file="${base_name}.wav"
  mkdir -p $1/$(dirname $audio_file)
  ffmpeg -loglevel fatal -y -i "$2" "$1/${audio_file}"
  echo "Converted $2 to $1/$audio_file"
' $directory $destination

echo "Conversion complete."
