## GAN 구현 및 실험

### Environment

$(pwd): GAN 폴더 경로로 대체할 것
```bash
$ docker build -t gan:v1.0 .
$ docker run -itd --name gan_env -v $(pwd):/workspace --gpus all gan:v1.0
$ docker exec -it gan_env bash
```