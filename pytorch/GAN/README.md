## GAN 구현 및 실험

Paper review: https://lowly-bearskin-8ac.notion.site/Generative-Adversarial-Nets-GAN-1ab1b30c9d1781169c42f2f99eaeb8f1?pvs=4

### Environment

$(pwd): GAN 폴더 경로로 대체할 것
```bash
$ docker build -t gan:v1.0 .
$ docker run -itd --name gan_env -v $(pwd):/workspace --gpus all gan:v1.0
$ docker exec -it gan_env bash
```

### Training
```bash
$ python train_gan.py --config configs/base_train_gan_config.yaml
$ python train_gan.py --config configs/base_train_dcgan_config.yaml
```

### Evaluation
```bash
$ python test_gan.py --config configs/base_train_gan_config.yaml
$ python test_gan.py --config configs/base_train_dcgan_config.yaml
```