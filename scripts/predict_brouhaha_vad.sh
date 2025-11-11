# set HF_TOKEN or HF_TOKEN_BROUHAHA
# HF_TOKEN_BROUHAHA should be used when models were received on different accaunts
# Prediction works only on one CUDA device

CUDA_VISIBLE_DEVICES=0 \
HF_TOKEN_BROUHAHA="YOUR HUGGINGFACE_TOKEN" \
HF_TOKEN="YOUR HUGGINGFACE_TOKEN" \
python -m VoicePersonification.main \
    -cp=../experiments/brouhaha_vad \
    -cn=predict
