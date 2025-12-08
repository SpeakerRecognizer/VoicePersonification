python -m grpc_tools.protoc \
  -Iservices/protos \
  --python_out=services/protos \
  --grpc_python_out=services/protos \
  services/protos/voice_personification.proto
  
# fix protobuf
sed -i 's/^import voice_personification_pb2 as voice__personification__pb2$/from services.protos import voice_personification_pb2 as voice__personification__pb2/' services/protos/voice_personification_pb2_grpc.py
