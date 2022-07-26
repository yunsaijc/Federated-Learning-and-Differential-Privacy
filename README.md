
### Parameter List

**Datasets**: MNIST, Cifar-10, FEMNIST, Fashion-MNIST, Shakespeare.

**Model**: CNN, MLP, LSTM for Shakespeare

**DP Mechanism**: Laplace, Gaussian(Simple Composition), **Todo**: Gaussian(*moments* accountant)

**DP Parameter**: $\epsilon$ and $\delta$

**DP Clip**: In DP-based FL, we usually clip the gradients in training and the clip is an important parameter to calculate the sensitivity.

### No DP
You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism no_dp


### Gaussian Mechanism
You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --dp_clip 10
