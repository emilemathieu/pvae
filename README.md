# [Hierarchal Representations with Poincaré Variational Auto-Encoders](https://arxiv.org/abs/1901.06033)

## Prerequisites
Modules in `requirements.txt`.

## Run experiments

### Synthetic dataset
```
CUDA_VISIBLE_DEVICES='' python3 pvae/main.py --model hyp_tree --latent-dim 2 --hidden-dim 200 --prior-std-scale 1.7 --data-dim 50 --data-params 6 2 1 1 5 5 --arch-dec Gyroplane --epochs 1000 --save-freq 1000 --lr 1e-3 --batch-size 64 --iwae-samples 5000
```

### MNIST dataset
```
CUDA_VISIBLE_DEVICES='' python3 pvae/main.py --model hyp_mnist --latent-dim 2 --hidden-dim 600 --c 0.7 --prior WrappedNormal --posterior WrappedNormal --arch-dec Gyroplane --arch-enc '' --lr 5e-4 --epochs 80 --save-freq 80 --batch-size 128 --iwae-samples 5000
```

## Running tests

```
pip3 install nose2
nose2
```

## References

If you find this code useful for your research, please cite the following paper in your publication:

```
@article{mathieu2019poincare,
  title={Hierarchical Representations with Poincar\'e Variational Auto-Encoders},
  author={Mathieu, Emile and Le Lan, Charline and Maddison, Chris J. and Tomioka, Ryota and Whye Teh, Yee},
  journal={arXiv preprint arXiv:1901.06033},
  year={2019}
}
```
