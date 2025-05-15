import os
import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="scp files creator",
        description="The program produces wav.scp and utt2spk for a given dataset",
    )
    parser.add_argument("--wav_path", help="Path to wavs", required=True)
    parser.add_argument("--scp_path", help="Path to directory with scp files", required=True)
    return parser.parse_args()


def create_scp_files(wav_path, scp_path):
    os.makedirs(scp_path, exist_ok=True)
    with open(f"{scp_path}/wav.scp", 'w') as wav_scp, open(f"{scp_path}/utt2spk", 'w') as utt2spk, open(f"{scp_path}/spk2utt", 'w') as spk2utt: 
        for spk in os.listdir(wav_path):
            for sess in os.listdir(f'{wav_path}/{spk}'):
                for utt in os.listdir(f'{wav_path}/{spk}/{sess}'):
                    wav_scp.write(f"{spk}/{sess}/{utt} {wav_path}/{spk}/{sess}/{utt}\n")
                    utt2spk.write(f"{spk}/{sess}/{utt} {spk}\n")
                    spk2utt.write(f"{spk} {spk}/{sess}/{utt}\n")


def main(wav_path, scp_path):
    create_scp_files(wav_path, scp_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.wav_path, 
         args.scp_path)