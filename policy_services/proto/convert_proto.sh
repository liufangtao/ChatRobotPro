
#进到本目录
cd "$(dirname "$0")"

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vla_server.proto

for x in ../server ../client;
do
cp *_pb2_grpc.py  ${x}/
cp *_pb2.py  ${x}/
done

rm *_pb2_grpc.py
rm *_pb2.py