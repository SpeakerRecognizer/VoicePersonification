import os
import torch
import math
import numpy as np
import torchaudio as ta

from torch import nn
from numpy.typing import NDArray
from typing import Dict, Iterable, Optional, Union
from itertools import repeat
from whisper.audio import (
    pad_or_trim,
    log_mel_spectrogram,
    N_SAMPLES,
    CHUNK_LENGTH,
    SAMPLE_RATE,
)
from huggingface_hub import hf_hub_download
from whisper.model import ResidualAttentionBlock, LayerNorm, Conv1d, sinusoids

from .verification_model import VerificationModel as BaseVerificationModel
from .wav2vec_bert import CurricularAAM


class Encoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(
            n_state, n_state, kernel_size=3, stride=2, padding=1
        )
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = nn.functional.gelu(self.conv1(x))
        x = nn.functional.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        nn.init.normal_(self.positional_embedding)  # ADDED

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor,
        kv_cache: Optional[dict] = None,
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)

        return x


class TimestampTokenizer:

    def __init__(self, max_duration: float, n_timestamp_tokens: int):
        self.max_duration = max_duration
        self.n_timestamp_tokens = n_timestamp_tokens
        self.spesials = ["sos", "eos"]

    def timestamp2token(self, timestamp: float):
        if (
            not 0
            <= timestamp
            <= self.max_duration * (1 + 1 / 2 / self.n_timestamp_tokens)
        ):
            raise ValueError(
                f"Invalid argument {timestamp=:.3f} for TimestampTokenizer {self.max_duration=:.2f}"
            )
        return int(
            min(1, timestamp / self.max_duration)
            * (self.n_timestamp_tokens - 1)
        )

    def token2timestamp(self, token: int):
        # if not 0 <= token < self.n_timestamp_tokens:
        #     raise ValueError(f"Invalid argument {token=} for TimestampTokenizer with {self.n_timestamp_tokens=}")
        return min(1, token / self.n_timestamp_tokens) * self.max_duration

    @property
    def sos_token(self):
        return self.n_timestamp_tokens + self.spesials.index("sos")

    @property
    def eos_token(self):
        return self.n_timestamp_tokens + self.spesials.index("eos")

    def _n_not_spesials(self):
        return self.n_timestamp_tokens

    @property
    def n_tokens(self):
        return len(self.spesials) + self._n_not_spesials()

    @property
    def n_cls_tokens(self):
        return self.n_timestamp_tokens + 2


class TimestampSpeakerTokeniser(TimestampTokenizer):

    def __init__(
        self, max_duration: float, n_timestamp_tokens: int, n_spks: int
    ):
        super().__init__(max_duration, n_timestamp_tokens)

        self.n_spks = n_spks

    def token2spk(self, token: int):
        if not 0 <= token - self.n_cls_tokens < self.n_spks:
            raise ValueError(f"Invalid argument {token=}")
        return token - self.n_cls_tokens

    def spk2token(self, spk: int):
        if not 0 <= spk < self.n_spks:
            raise ValueError(f"Invalid argument {spk=}")
        return spk + self.n_cls_tokens

    def _n_not_spesials(self):
        return super()._n_not_spesials() + self.n_spks


def crop_timestamps(
    ts: Union[NDArray, torch.Tensor],
    start: float,
    stop: float,
    min_seg: float = 0.1,
):
    ts = ts[(ts[..., 0] < stop) | (ts[..., 1] > start)]

    if ts.shape[0] == 0:
        return ts

    if ts[0, 0] < start:
        ts[0, 0] = start
    if ts[-1, -1] > stop:
        ts[-1, -1] = stop
    if ts[0, 1] - ts[0, 0] < min_seg:
        ts = ts[1:]
    if ts.shape[0] != 0 and ts[-1, 1] - ts[-1, 0] < min_seg:
        ts = ts[:-1]

    return ts


class VerificationSegmentstionModel(nn.Module):

    def __init__(
        self,
        n_mels,
        n_audio_ctx,
        n_audio_state,
        n_audio_head,
        n_audio_layer,
        n_vocab,
        n_text_ctx,
        n_text_state,
        n_text_head,
        n_text_layer,
        n_spks: int,
        n_timestamps: int = 1500,
        chunk_duration: float = 30.0,
    ):
        super().__init__()

        self.encoder = Encoder(
            n_mels,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
        )
        self.decoder = Decoder(
            n_vocab,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
        )

        self.tokenizer = TimestampSpeakerTokeniser(
            chunk_duration, n_timestamps, n_spks
        )

        self.ts_head = nn.Linear(n_text_state, self.tokenizer.n_cls_tokens)
        self.spk_head = CurricularAAM(n_text_state, self.tokenizer.n_spks)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def decode_featmap(
        self, tokens: torch.Tensor, audio_features: torch.Tensor
    ):
        return self.decoder(tokens, audio_features)

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
        embeded_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        if embeded_audio is None:
            embeded_audio = self.embed_audio(mel)

        output = self.decode(tokens, embeded_audio)

        return output

    def build_tokens_mask(self, n_tokens: torch.LongTensor):
        len_mask = torch.arange(n_tokens.max()) < n_tokens.unsqueeze(1).cpu()
        ts_mask = len_mask.clone()
        ts_mask[..., 2::3] = False
        spk_mask = ~ts_mask & len_mask

        return ts_mask, spk_mask

    def segemntate(
        self,
        wav: Union[NDArray, torch.Tensor],
        sample_rate: int = SAMPLE_RATE,
        init_timestamps: Optional[Union[NDArray, torch.Tensor]] = None,
    ):

        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)

        if isinstance(init_timestamps, np.ndarray):
            init_timestamps = torch.from_numpy(init_timestamps)

        if sample_rate != SAMPLE_RATE:
            wav = ta.functional.resample(wav, sample_rate, SAMPLE_RATE)

        wav_chunks = torch.chunk(
            wav,
            math.ceil(
                wav.shape[-1] / self.tokenizer.max_duration / SAMPLE_RATE
            ),
        )

        if wav_chunks[-1].shape[-1] < 1 * SAMPLE_RATE:
            wav_chunks = wav_chunks[:-1]

        embs = []
        timestamps = []

        for i, wav_chunk in enumerate(wav_chunks):
            init_ts_chunk = crop_timestamps(
                init_timestamps, i * CHUNK_LENGTH, (i + 1) * CHUNK_LENGTH
            ) - (i * CHUNK_LENGTH)
            embs_chunk, timestamps_chunk = self.segmentate_chunk(
                wav_chunk, init_ts_chunk
            )
            embs.append(embs_chunk)
            timestamps.append(timestamps_chunk + (i * CHUNK_LENGTH))

        return torch.concat(embs), torch.concat(timestamps).to(wav.device)

    def segmentate_chunk(
        self, wav: torch.Tensor, init_timestamps: Optional[torch.Tensor] = None
    ):
        wav = pad_or_trim(wav)
        mel = log_mel_spectrogram(wav)
        tokens = [self.tokenizer.sos_token]

        spk_embs = []
        timestamps = []
        to_tensor = lambda t: torch.LongTensor(t).view(1, -1).to(wav.device)
        init_timestamps = (
            repeat(None) if init_timestamps is None else init_timestamps
        )

        with torch.no_grad():
            embeded_audio = self.embed_audio(mel.unsqueeze(0))

            for ts in init_timestamps:
                if ts is None:
                    ts_start = self.tokenizer.timestamp2token(ts[0].item())
                else:
                    decoder_output = self.decode_featmap(
                        to_tensor(tokens), embeded_audio
                    )
                    ts_start = self.tsemb2token(decoder_output[0, -1])

                if ts_start == self.tokenizer.eos_token:
                    break

                tokens.append(ts_start)
                if ts is None:
                    ts_end = self.tokenizer.timestamp2token(ts[1].item())
                else:
                    decoder_output = self.decode_featmap(
                        to_tensor(tokens), embeded_audio
                    )
                    ts_end = self.tsemb2token(decoder_output[0, -1])

                tokens.append(ts_end)
                decoder_output = self.decode_featmap(
                    to_tensor(tokens), embeded_audio
                )
                spk_token = self.spkemb2token(decoder_output[0, -1])

                tokens.append(spk_token)
                spk_embs.append(decoder_output[0, -1])
                timestamps.append(
                    [
                        self.tokenizer.token2timestamp(ts_start),
                        self.tokenizer.token2timestamp(ts_end),
                    ]
                )
        spk_embs = (
            torch.vstack(spk_embs)
            if spk_embs
            else torch.Tensor([]).to(wav.device)
        )

        return spk_embs, torch.Tensor(timestamps)

    def spkemb2token(self, emb: torch.Tensor):
        return self.tokenizer.spk2token(
            (emb @ self.spk_head.kernel).argmax()
        ).item()

    def tsemb2token(self, emb: torch.Tensor):
        return self.ts_head(emb).argmax().item()

    @staticmethod
    def from_pretrained(
        name_or_path: str = "itmo_personification_model_segmentation.ckpt", 
        repo: str = "VoicePersonificationITMO/NIRSIModels"
    ):
        if not os.path.isfile(name_or_path):
            name_or_path = hf_hub_download(repo_id=repo, filename=name_or_path)

        checkpoint = torch.load(name_or_path, map_location="cpu")
        model = VerificationSegmentstionModel(**checkpoint["args"])
        model.load_state_dict(checkpoint["state_dict"])

        return model


class ITMOPersonificationModelSegmentation(BaseVerificationModel):

    def __init__(self, model_name_or_path: str = "itmo_personification_model_segmentation.ckpt"):
        super().__init__()
        self.model = VerificationSegmentstionModel.from_pretrained(
            model_name_or_path
        )

    def forward(self, wavs: torch.Tensor):

        wavs -= torch.mean(wavs, dim=-1)
        wavs /= torch.max(wavs, dim=-1).values

        if wavs.shape[0] != 1:
            raise ValueError(
                f"{ITMOPersonificationModelSegmentation.__name__} accept validation batch size == 1 only"
            )

        init_timestamps = torch.Tensor([[0, wavs.shape[-1] / SAMPLE_RATE]])

        spk_embs, _ = self.model.segemntate(
            wavs.squeeze(0), init_timestamps=init_timestamps
        )
        emb = torch.mean(spk_embs, dim=0)

        return emb.unsqueeze(0)
