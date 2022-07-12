# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F  # 加载nn中的功能函数
import torch.optim as optim  # 加载优化器有关包
import torch.utils.data as Data
from torchvision import datasets, transforms  # 加载计算机视觉有关包
from torch.autograd import Variable
import syft as sy
import codecs
import utils

import copy
import math
import random
import datetime

hook = sy.TorchHook(torch)


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        # CNN or LR or SIMPLE
        self.model_type = "CNN"


args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cpu")
# device = torch.device("cuda" if use_cuda else "cpu")


def get_label_clusters(pattern_idx):
    if pattern_idx == 0:  # IID
        label_clusters = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    elif pattern_idx == 1:  # lowbias
        label_clusters = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
    elif pattern_idx == 2:  # midbias
        label_clusters = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    elif pattern_idx == 3:  # highbias
        label_clusters = ((0,), (1,), (2,), (3,), (4,),
                          (5,), (6,), (7,), (8,), (9,))
    return label_clusters


def distributing_data(args, worker_num, train_loader, pattern_idx):
    worker_list = []
    worker = {}
    for i in range(1, worker_num + 1):
        worker[str(i)] = sy.VirtualWorker(hook, id='worker' + str(i))
        worker_list.append(str(i))
    # print(locals()['worker'+str(i)]._objects)
    secure_worker = sy.VirtualWorker(hook, id="secure_worker")
    server = sy.VirtualWorker(hook, id="server")

    # 分割训练任务
    print(len(train_loader), len(list(enumerate(train_loader))))

    train_data = {}
    train_target = {}

    for i in range(1, worker_num + 1):
        train_data[str(i)] = []
        train_target[str(i)] = []

    # for X,label in test_dataset:
    #   # for i,(X,label) in enumerate(test_loader):
    #   X = X.view(-1,784)
    #   X = Variable(X)
    #   label = torch.tensor([label])
    #   data_set.append(X.numpy().tolist())
    #   target_set.append(label.numpy().tolist())
    #   train_data[str(i % worker_num + 1)].append(X.numpy().tolist())
    #   train_target[str(i % worker_num + 1)].append(label.numpy().tolist())

    if pattern_idx == 0:
        for i, (X, label) in enumerate(train_loader):
            if args.model_type == "LR" or args.model_type == "SIMPLE":
                X = X.view(-1, 784)
            else:
                pass
            X = Variable(X)
            label = Variable(label)
            # print("(X,label)",(X,label))
            # print("label",label)
            if len(X) != args.batch_size:
                continue
            train_data[str(i % worker_num + 1)].append(X.numpy().tolist())
            train_target[str(i % worker_num + 1)].append(label.numpy().tolist())
    else:
        label_clusters = get_label_clusters(pattern_idx)
        clusters_num = len(label_clusters)
        # clusters_data_num = {1:num_1,2:num_2,...}
        clusters_data_num = {}
        clusters_worker_num = {}
        for i in range(1, worker_num + 1):
            cluster = i % clusters_num
            clusters_worker_num.setdefault(cluster, [])
            clusters_worker_num[cluster].append(i)

        for i, (X, label) in enumerate(train_loader):
            # find worker to distribute
            a_label = label.numpy().tolist()[0]
            for j in range(len(label_clusters)):
                if a_label in label_clusters[j]:
                    cluster = j
                    break
            clusters_data_num.setdefault(cluster, -1)
            clusters_data_num[cluster] += 1
            worker_order = clusters_data_num[cluster] % len(clusters_worker_num[cluster])
            choose_worker = clusters_worker_num[cluster][worker_order]
            # distribute
            if args.model_type == "LR" or args.model_type == "SIMPLE":
                X = X.view(-1, 784)
            else:
                pass
            X = Variable(X)
            label = Variable(label)
            if len(X) != args.batch_size:
                continue
            train_data[str(choose_worker)].append(X.numpy().tolist())
            train_target[str(choose_worker)].append(label.numpy().tolist())
        # train_data[str(i % worker_num + 1)].append(X.numpy().tolist())
        # train_target[str(i % worker_num + 1)].append(label.numpy().tolist())
    print(clusters_worker_num)
    print(clusters_data_num)

    print('here!')

    worker_data = {}
    worker_target = {}
    for i in range(1, worker_num + 1):
        train_data[str(i)] = torch.tensor(train_data[str(i)])
        train_target[str(i)] = torch.tensor(train_target[str(i)])
        worker_data[str(i)] = train_data[str(i)].send(worker[str(i)])
        worker_target[str(i)] = train_target[str(i)].send(worker[str(i)])

    print('here!here!')

    return worker, secure_worker, server, worker_data, worker_target


def add_model(dst_model, src_model):
    """Add the parameters of two models.
	Args:
		dst_model (torch.nn.Module): the model to which the src_model will be added.
		src_model (torch.nn.Module): the model to be added to dst_model.
	Returns:
		torch.nn.Module: the resulting model of the addition.
	"""
    params1 = dst_model.state_dict().copy()
    params2 = src_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] + params2[name1]
    model = copy.deepcopy(dst_model)
    model.load_state_dict(params1, strict=False)
    return model


def scale_model(model, scale):
    """Scale the parameters of a model.
	Args:
		model (torch.nn.Module): the models whose parameters will be scaled.
		scale (float): the scaling factor.
	Returns:
		torch.nn.Module: the module with scaled parameters.
	"""
    params = model.state_dict().copy()
    # scale = torch.tensor(scale)
    with torch.no_grad():
        for name in params:
            # params[name] = params[name].type_as(scale) * scale
            params[name] = params[name] * scale
    scaled_model = copy.deepcopy(model)
    scaled_model.load_state_dict(params, strict=False)
    return scaled_model


def aggregate_model(server_model, update_model, yita):
    params1 = server_model.state_dict().copy()
    params2 = update_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = (1 - yita) * params1[name1] + yita * params2[name1]
    model = copy.deepcopy(server_model)
    model.load_state_dict(params1, strict=False)
    return model


def model_test(args, model, runtime, train_mode, alpha, yita):
    loss_f = nn.CrossEntropyLoss()
    model.eval()
    now_loss = 0
    correct = 0

    model = model.get()
    for i, (X, label) in enumerate(test_loader):
        X = X.to(device)
        label = label.to(device)
        if args.model_type == "LR" or args.model_type == "SIMPLE":
            X = X.view(-1, 784)
        else:
            pass
        X = Variable(X)
        testout = model(X)
        testloss = loss_f(testout, label)
        # testloss = F.nll_loss(testout, label)
        # if args.model_type == "LR" or args.model_type=="SIMPLE":
        # 	loss_f = nn.CrossEntropyLoss()#损失函数选择，交叉熵函数
        # 	testloss = loss_f(out, label)
        pred = testout.argmax(1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        now_loss += float(testloss)
    now_loss /= len(test_loader)
    print(now_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        now_loss, correct, len(test_loader) * args.batch_size,
                           100. * correct / (len(test_loader) * args.batch_size)))

    f = open(args.model_type + "_" + train_mode + "_" + str(yita) + "_" + str(alpha) + "_" + str(
        args.batch_size) + "_" + str(pattern_idx) + "_" + str(args.lr) + "_" + str(worker_num) + ".txt", 'a')
    newline = str(now_loss) + "	" + str(correct / (len(test_loader) * args.batch_size)) + "	" + str(runtime)
    f.writelines(newline + "\n")
    f.close()


def get_prepare_time(worker_num):
    lower_bound = 10
    upper_bound = 100
    prepare_time = {}

    now_value = lower_bound
    difference_value = (upper_bound - lower_bound) / (worker_num - 1)
    for i in range(1, worker_num + 1):
        prepare_time[str(i)] = now_value + random.uniform(-0.0001, 0.0001)
        now_value += difference_value

    return prepare_time


# current_lost_time = {"1":,3;"2":4,...}
def scheduler(choose_num, prepare_time, current_lost_time):
    current_lost_time_order = sorted(current_lost_time.items(), key=lambda x: x[1], reverse=False)
    choose_workers = []

    for i in range(0, choose_num):
        choose_worker = current_lost_time_order[i][0]
        iteration_time = current_lost_time_order[i][1]
        choose_workers.append(choose_worker)
    iteration_time = current_lost_time_order[choose_num - 1][1]

    for worker in list(current_lost_time.keys()):
        if current_lost_time[worker] <= iteration_time:
            current_lost_time[worker] = prepare_time[worker]
        else:
            current_lost_time[worker] = current_lost_time[worker] - iteration_time

    return choose_workers, current_lost_time, iteration_time


def hybrid_update(model, worker_num, worker, secure_worker, server, worker_data, worker_target, test_loader, alpha,
                  yita):
    model.train()                   # 设置模型为训练模式
    loss_f = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数

    print("yita=", yita)
    print("alpha=", alpha)

    choose_num = int(alpha * worker_num)            # 文章中的M值
    prepare_time = get_prepare_time(worker_num)     # 模型准备时间
    print("prepare_time:", prepare_time)

    worker_model = {}
    for i in range(1, worker_num + 1):
        worker_model[str(i)] = model.copy().send(worker[str(i)])

    worker_opt = {}                                 # worker的优化器
    for i in range(1, worker_num + 1):
        worker_opt[str(i)] = optim.SGD(params=worker_model[str(i)].parameters(), lr=args.lr)

    current_lost_time = copy.deepcopy(prepare_time)

    learning_rate = args.lr
    runtime = 0
    tau = {}                    # 模型延迟
    tau_th = 999                # 模型延迟的阈值
    print("tau_th =", tau_th)
    count = {}
    for i in range(1, worker_num + 1):
        tau[str(i)] = 0
        count[str(i)] = 0
    # for iteration in range(1000):
    # while 1:
    while runtime < 10000:
        print("tau =", tau)
        server_model = model.copy().send(server)
        (choose_workers, current_lost_time, iteration_time) = scheduler(choose_num, prepare_time, current_lost_time)
        # choose_workers = random.sample(range(1,len(worker)+1),choose_num)
        print("choose_workers:", choose_workers)
        runtime += iteration_time
        print("runtime=", runtime)

        # 本地更新
        for choose_worker in choose_workers:
            count[str(choose_worker)] += 1
            for wi in range(1):
                for j in range(0, len(worker_data[str(choose_worker)])):
                    worker_opt[str(choose_worker)].zero_grad()
                    out = worker_model[str(choose_worker)](worker_data[str(choose_worker)][j].to(device))
                    label = worker_target[str(choose_worker)][j].to(device)
                    lossvalue = loss_f(out, label)
                    lossvalue.backward()  # 反向转播，刷新梯度值
                    worker_opt[str(choose_worker)].step()
        count_total = sum(count.values())
        f = {}
        for choose_worker in choose_workers:
            f[str(choose_worker)] = count[str(choose_worker)] / count_total
        print("f:", f)

        # 把模型移动到secure_worker做简单平均
        for choose_worker in choose_workers:
            worker_model[str(choose_worker)].move(secure_worker)
        server_model.move(secure_worker)

        # 全局聚合，舍弃stale模型
        update_worker_num = 0
        with torch.no_grad():
            for choose_worker in choose_workers:
                if tau[str(choose_worker)] > tau_th:
                    continue
                if update_worker_num == 0:
                    update_model = worker_model[str(choose_worker)].copy()
                else:
                    update_model = add_model(update_model, worker_model[str(choose_worker)])
                update_worker_num += 1
            if update_worker_num != 0:
                update_model = scale_model(update_model, 1.0 / update_worker_num)
                model = aggregate_model(server_model, update_model, update_worker_num / worker_num)
                del update_model
            else:
                model = scale_model(server_model, 1)

        model_test(args, model, runtime, "hybrid_tau" + str(tau_th), alpha, yita)

        # 把模型复制到worker
        for choose_worker in choose_workers:
            frequency = f[str(choose_worker)]
            # r = math.log(2+tau[choose_worker],2)
            # r = temp_ratio[str(choose_worker)]
            # r = 1
            true_learning_rate = learning_rate / (worker_num * frequency)
            true_learning_rate = max(0.001, true_learning_rate)
            worker_model[str(choose_worker)] = model.copy().send(worker[str(choose_worker)])
            worker_opt[str(choose_worker)] = optim.SGD(params=worker_model[str(choose_worker)].parameters(),
                                                       lr=true_learning_rate)

        # 更新模型延迟
        for i in range(1, worker_num + 1):
            if str(i) in choose_workers:
                tau[str(i)] = 0
            else:
                tau[str(i)] += 1


if __name__ == "__main__":
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print("Loading data...")
    pattern_idx = 1
    print("pattern_idx =", pattern_idx)
    # (train_dataset, test_dataset) = load_data()

    # print("Building data distribution...")
    # BATCH_SIZE = 64
    print("BATCH_SIZE =", args.batch_size)

    worker_num = 10
    # model = nn.Linear(784,10).to(device)
    if args.model_type == "CNN":
        model = utils.MNIST_CNN_Net().to(device)
    elif args.model_type == "LR":
        model = utils.MNIST_LR_Net().to(device)
    elif args.model_type == "SIMPLE":
        model = nn.Linear(784, 10).to(device)
    print("Model is", args.model_type)

    print("Distributing data to workers...")
    if 0 < pattern_idx < 1:
        train_loader_1 = torch.load("train_loader_" + str(pattern_idx) + "1_" + str(args.batch_size) + ".pt")
        print("len train_loader_1=", len(train_loader_1))
        train_loader_2 = torch.load("train_loader_" + str(pattern_idx) + "2_" + str(args.batch_size) + ".pt")
        print("len train_loader_2=", len(train_loader_2))
        test_loader = torch.load("test_loader_" + str(1) + "_" + str(args.batch_size) + ".pt")
        # (worker, secure_worker, server, worker_data, worker_target) = distributing_data_2(args, worker_num,
        #                                                                                   train_loader_1,
        #                                                                                   train_loader_2, pattern_idx)
    else:
        train_loader = torch.load("train_loader_" + str(pattern_idx) + "_" + str(args.batch_size) + ".pt")
        test_loader = torch.load("test_loader_" + str(pattern_idx) + "_" + str(args.batch_size) + ".pt")
        (worker, secure_worker, server, worker_data, worker_target) = distributing_data(args, worker_num, train_loader,
                                                                                        pattern_idx)

    print("混合更新:")
    for yita in [0.5]:
        alpha = yita
        hybrid_update(model, worker_num, worker, secure_worker, server, worker_data, worker_target, test_loader, alpha,
                      yita)
        print("hybrid_update" + str(yita))

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
