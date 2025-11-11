
export HF_TOKEN=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

CUDA_VISIBLE_DEVICES=0,1,2 \
python -m VoicePersonification.main \
    -cp=../experiments/itmo-personification-model-segmentation \
    -cn=test