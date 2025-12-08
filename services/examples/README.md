# Quick start

* generate protobuf
```python 
bash services/protos/generate_protoc.sh
```
export HF_TOKEN=<Your Hugging Face token>
export HF_TOKEN_BROUHAHA=<Your Hugging Face token>

* run services
~~~bash
~$ python -m services.brouhaha_vad_service --port 50051
~$ python -m services.ecapa_tdnn_service --port 50052
~$ python -m services.itmo_personification_model_large_service --port 50053
~$ python -m services.itmo_personification_model_segmentation_service --port 50054
~$ python -m services.whisper_recognition_service --port 50055
~~~

* run tests
~~~bash
~$ python -m services.examples.multi_service_client ./examples/id10277-znxUWA2QAGs-00001.wav

~$python -m services.examples.ecapa_tdnn ./examples/id10277-znxUWA2QAGs-00001.wav
~~~
