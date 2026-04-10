# docker-action-test-cuda

convert to onnx to tensorrt

```
python3 export_yolo26.py -w hazmat_yolo26.pt --opset 18
/usr/src/tensorrt/bin/trtexec --onnx=hazmat_yolo26.onnx --saveEngine=hazmat_yolo26.engine --fp16 --verbose
```

build
```
docker build -t cuda_test:dev .
docker build -t docker_upgrade_test:dev -f dockerfile.test .
```

run
```
```