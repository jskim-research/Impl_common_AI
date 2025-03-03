## GAN 구현 및 실험

Paper review: https://lowly-bearskin-8ac.notion.site/Auto-Encoding-Variational-Bayes-1ab1b30c9d178174ac38e90463037a62?pvs=4

### Environment

$(pwd): VAE 폴더 경로로 대체할 것
```bash
$ docker build -t vae:v1.0 .
$ docker run -itd --name vae_env -v $(pwd):/workspace --gpus all vae:v1.0
$ docker exec -it vae_env bash
```

### Training
```bash
$ python train_vae.py --config configs/base_train_vae_config.yaml
```

### Evaluation
```bash
$ python test_vae.py --config configs/base_train_vae_config.yaml
```