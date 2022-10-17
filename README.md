
### No DP
You can run like this:

python FedSA.py --dataset mnist --iid --model cnn --epochs 1000 --dp_mechanism no_dp


### Gaussian Mechanism
You can run like this:

python FedSA.py --dataset mnist --iid --model cnn --epochs 1000 --dp_mechanism Gaussian --dp_epsilon 10
