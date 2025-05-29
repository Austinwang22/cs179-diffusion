Austin Wang, Bobby Wang

For our CS 179 project, we are implementing diffusion models (both training and sampling). We are following the framework
presented in [this paper](https://arxiv.org/pdf/2206.00364).

For our CPU implementation, check out the pyref directory. We implemented the training and inference pipelines using PyTorch.
We also implemented a UNet with PyTorch. Our code can be run both with CPU or GPU (simply set the device argument when running a file). 

To train a diffusion model on MNIST, run the following:

```bash
python3 pyref/train.py
```

To generate new samples, run:

```bash
python3 pyref/sample.py
```

Here are some generated samples:

![Alt text](figs/sample2.png)
![Alt text](figs/sample3.png)
![Alt text](figs/sample4.png)