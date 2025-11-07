# Quick start

* generate protobuf
```python 
bash services/protos/generate_protoc.sh
```

* run services
~~~bash
~$ python -m services.brouhaha_vad_service --port 50051
~$ python -m services.ecapa_tdnn_service --port 50052
~$ python -m services.itmo_personification_model_large_service --port 50053
~$ python -m services.itmo_personification_model_segmentation_service --port 50054
~~~

* run tests
~~~bash
~$ python -m services.examples.brouhaha_vad /Users/nikitossii/Documents/VoicePersonification/examples/id10277-znxUWA2QAGs-00001.wav

~$python -m services.examples.ecapa_tdnn /Users/nikitossii/Documents/VoicePersonification/examples/id10277-znxUWA2QAGs-00001.wav
~~~
