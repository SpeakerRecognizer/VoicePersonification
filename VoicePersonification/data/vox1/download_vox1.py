from VoicePersonification.data.vox1.download_utils import download_dataset, download_protocol, concatenate, extract_dataset

def main():
    # Download VoxCeleb1 (test set)
    with open('data/lists/datasets.txt', 'r') as f:
        lines = f.readlines()
    download_dataset(lines, user='voxceleb1902', password='nx0bl2v2', save_path='data')

    # Download VoxCeleb1 identification protocol
    with open('data/lists/protocols.txt', 'r') as f:
        lines = f.readlines()
        
    download_protocol(lines, save_path='data/voxceleb1_test')

    # Concatenate archives for VoxCeleb1 dev set
    with open('data/lists/concat_arch.txt', 'r') as f:
        lines = f.readlines()        
    concatenate(lines, save_path='data')

    # Extract VoxCeleb1 dev set
    extract_dataset(save_path='data/voxceleb1_dev', fname='data/vox1_dev_wav.zip')


if __name__ == '__main__':
    main()