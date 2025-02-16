import numpy as np


def _parese_wav_scp_line(line: list):
    if len(line) == 2:
        return line
    else:
        path = line[2]
        if path[0] == '"' and path[-1] == '"':
            path = path[1:-1]
        return line[0], path

def read_scp(fpath: str, sep: str = " "):
    with open(fpath, "r") as if_:
        lines = [l.strip().split(sep) for l in if_ if l.strip() != ""]
    val_dct = {k: v for k, v in map(_parese_wav_scp_line, lines)}

    return val_dct

def apply_vad(wav: np.array, vad: np.array):
    n_repeats = np.ceil(wav.shape[-1] / vad.shape[-1])
    vad = np.repeat(vad.astype(np.bool_), n_repeats)[..., :wav.shape[-1]]

    return wav[vad]