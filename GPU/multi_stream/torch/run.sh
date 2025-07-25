rm torch_multi_stream.nsys-rep

nsys profile -t cuda,cudnn,cublas -o torch_multi_stream python3 multi_stream.py