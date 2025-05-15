#!/bin/bash  

download_dir=$1
rawdata_dir=$2
  
echo "Decompress all archives ..."
echo "This could take some time ..."

for archive in vox1_test_wav.zip vox1_dev_wav.zip vox2_aac.zip; do
[ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
done
[ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

if [ ! -d ${rawdata_dir}/voxceleb1 ]; then
mkdir -p ${rawdata_dir}/voxceleb1/test ${rawdata_dir}/voxceleb1/dev
unzip ${download_dir}/vox1_test_wav.zip -d ${rawdata_dir}/voxceleb1/test
unzip ${download_dir}/vox1_dev_wav.zip -d ${rawdata_dir}/voxceleb1/dev
fi

if [ ! -d ${rawdata_dir}/voxceleb2_m4a ]; then
mkdir -p ${rawdata_dir}/voxceleb2_m4a
unzip ${download_dir}/vox2_aac.zip -d ${rawdata_dir}/voxceleb2_m4a
fi

echo "Decompress success !!!"
