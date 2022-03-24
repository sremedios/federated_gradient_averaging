# Federated Gradient Averaging
This code is associated with the paper "Federated Gradient Averaging for Multi-Site Training with Momentum-Based Optimizers".

# How to use:
Clone this repository on all participating sites as well as on the central server. Ensure all sites have ssh access to the central server.

On the server, run `server.py`, passing in as an argument the dataset to use, either `CT` or `MNIST`.

On each client, run one of `train_ct.py`, `train_mnist.py`, `test_ct.py`, `test_mnist.py`, depending on the experiment. All clients must run the same script; other combinations will result in undefined behavior.

Valid `MODE` arguments include `cyclic`, `federated`, `weightavg`, `local`.

An example command line argument to train MNIST with cyclic weight transfer using GPU 0 and port 10203:

`python train_ct.py A cyclic 0 /path/to/training/set/ 10203`

# Citation
Please consider citing our relevant papers:
```
@article{remedios2020FederatedGradientAveraging,
  title={Federated Gradient Averaging for Multi-Site Training with Momentum-Based Optimizers},
  author={Remedios, Samuel W and Butman, John A and Landman, Bennett A and Pham, Dzung L},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2020},
  organization={Springer}
}

@article{remedios2020DistributedDeepLearning,
  title={Distributed deep learning across multisite datasets for generalized CT hemorrhage segmentation},
  author={Remedios, Samuel W and Roy, Snehashis and Bermudez, Camilo and Patel, Mayur B and Butman, John A and Landman, Bennett A and Pham, Dzung L},
  journal={Medical physics},
  volume={47},
  number={1},
  pages={89--98},
  year={2020},
  publisher={Wiley Online Library}
}
```
