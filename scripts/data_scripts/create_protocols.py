import os
import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="protocol creator",
        description="The program produces imp-enroll-test.txt and tar-enroll-test.txt protocols",
    )
    parser.add_argument("--trials_path", help="Path to vox1-O(cleaned).txt", required=True)
    parser.add_argument("--protocols_path", help="Desired path to protocols", required=True)
    return parser.parse_args()


def create_protocols(trials_path, protocols_path):
    os.makedirs(protocols_path, exist_ok=True)
    with open(f"{trials_path}", 'r') as trials, open(f"{protocols_path}/tar-enroll-test.txt", 'w') as tar, open(f"{protocols_path}/imp-enroll-test.txt", 'w') as imp: 
        for line in trials:
            kind, enroll, test = line.strip().split(' ')
            if kind == '0':
                imp.write(f'{enroll} {test}\n')
            if kind == '1':
                tar.write(f'{enroll} {test}\n')    


def main(trials_path, protocols_path):
    create_protocols(trials_path, protocols_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.trials_path, 
         args.protocols_path)