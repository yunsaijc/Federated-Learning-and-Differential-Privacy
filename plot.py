import os
import sys

import matplotlib.pyplot as plt
color = ['black', 'brown', 'green', 'blue', 'red']

def get(file, limit=-1):
    """
    Read log file and return 3 lists: runtime, accuracy, loss
    Args:
        file: File path
        limit: The boundary of plot

    Returns: running_time, accuracy, loss

    """
    file.seek(0)
    x, y, z = [], [], []    # runtime, acc, loss
    if limit != -1:     # 读到特定的某一行
        for i in range(limit):
            tmpLine = file.readline().strip().replace('\n', ' ').replace('\t', ' ')
            lst = tmpLine.split()
            tmpx, tmpy, tmpz = lst[0], lst[1], lst[2]
            x.append(round(eval(tmpx), 3))
            y.append(round(eval(tmpy), 3))
            z.append(round(eval(tmpz), 3))
    else:               # 读到最后一行
        lineNum = 1
        while file.readable():
            tmpLine = file.readline().strip().replace('\n', ' ').replace('\t', ' ')
            # print(tmpLine)
            if len(tmpLine) == 0:
                break
            lst = tmpLine.split()
            # print(lst)
            tmpx, tmpy, tmpz = lst[0], lst[1], lst[2]
            x.append(round(eval(tmpx), 3))
            y.append(round(eval(tmpy), 3))
            z.append(round(eval(tmpz), 3))
            lineNum += 1
            # if round(eval(tmpx), 3) > 5000:
            #     break
    return x, y, z


def get_tau(file, limit):
    """
    Read log file and return 3 lists: round, tau, T_avg
    Args:
        file: File path
        limit: The boundary of plot

    Returns: running_time, accuracy, loss

    """
    file.seek(0)
    x, y, z = [], [], []    # rnd, tau, T_avg
    if limit != -1:     # 读到特定的某一行
        rnd = 1
        for i in range(limit):
            tmpLine = file.readline().strip().replace('\n', ' ').replace('\t', ' ')
            lst = tmpLine.split()
            tmpx, tmpy, tmpz = lst[0], lst[1], lst[2]
            x.append(rnd)
            y.append(round(eval(tmpy), 3))
            z.append(round(eval(tmpz), 3))
            rnd += 1
    else:               # 读到最后一行
        lineNum = 1
        while file.readable():
            tmpLine = file.readline().strip().replace('\n', ' ').replace('\t', ' ')
            # print(tmpLine)
            if len(tmpLine) == 0:
                break
            lst = tmpLine.split()
            # print(lst)
            tmpx, tmpy, tmpz = lst[0], lst[1], lst[2]
            x.append(lineNum)
            y.append(round(eval(tmpy), 3))
            z.append(round(eval(tmpz), 3))
            lineNum += 1
    return x, y, z


def plotFig(model: str, dataset: str, title: str, iid: bool, mode: str, savePath: str, files: list, labels: list):
    """

    Args:
        model: The model that is trained
        dataset: The dataset used to train
        title: Title of the figure
        iid: IID data or not
        mode: Plot the figure of "acc" or "loss"
        savePath: The path to save the figure
        files: Data files
        labels: Labels on the figure

    Returns:

    """
    # basic settings
    linewidth = 0.5
    plt.figure(dpi=400, figsize=(16, 8))
    plt.xlabel('Running time (s)')

    plt.title(title)
    x, y, z = [], [], []
    for i in range(len(files)):
        tx, ty, tz = get(files[i], -1)
        x.append(tx)
        y.append(ty)
        z.append(tz)

    if mode == 'acc':
        for i in range(len(x)):
            x[i].insert(0, 0)
            y[i].insert(0, 0)
        plt.ylabel('Accuracy (%)')
        for i in range(len(x)):
            plt.plot(x[i], y[i], color=color[i], label=labels[i], linewidth=linewidth)
        str_1 = "acc"
    elif mode == 'loss':
        plt.ylabel('Loss')
        for i in range(len(x)):
            plt.plot(x[i], z[i], color=color[i], label=labels[i], linewidth=linewidth)
        str_1 = "loss"
    else:
        sys.exit(1)
    plt.legend()
    if iid:
        str_2 = "iid"
    else:
        str_2 = "N-iid"
    plt.savefig(savePath + str_1 + "_" + str_2 + "_" + model + "_" + dataset + ".png")


def plotFigWithoutFile(title: str, savePath: str, filename: str, x: list, y: list, xlabel: str, ylabel: str, labels: list):
    # basic settings
    linewidth = 0.5
    plt.figure(dpi=400, figsize=(16, 8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for i in range(len(y)):
        plt.plot(x[0], y[i], color=color[i], label=labels[i], linewidth=linewidth, marker="o")
    plt.legend()
    plt.savefig(savePath + filename)


def plotFig_tau(model: str, dataset: str, title: str, iid: bool, mode: str, savePath: str, files: list, labels: list):
    """

    Args:
        model: The model that is trained
        dataset: The dataset used to train
        title: Title of the figure
        iid: IID data or not
        mode: Plot the figure of "acc" or "loss"
        savePath: The path to save the figure
        files: Data files
        labels: Labels on the figure

    Returns:

    """
    # basic settings
    linewidth = 1
    plt.figure(dpi=400, figsize=(16, 8))
    plt.xlabel('Round')

    plt.title(title)
    x, y, z = [], [], []
    for i in range(len(files)):
        tx, ty, tz = get(files[i], 300)
        x.append(tx)
        y.append(ty)
        z.append(tz)

    if mode == 'tau':
        for i in range(len(x)):
            x[i].insert(0, 0)
            y[i].insert(0, 0)
        plt.ylabel(r'$\tau_{max}$')
        for i in range(len(x)):
            plt.plot(x[i], y[i], color=color[i], label=labels[i], linewidth=linewidth)
        str_1 = "tau"
    elif mode == 't':
        plt.ylabel(r'$\overline{t}(s)$')
        for i in range(len(x)):
            plt.plot(x[i], z[i], color=color[i], label=labels[i], linewidth=linewidth)
        str_1 = "t"
    else:
        sys.exit(1)
    plt.legend()
    if iid:
        str_2 = "iid"
    else:
        str_2 = "N-iid"
    plt.savefig(savePath + str_1 + "_" + str_2 + "_" + model + "_" + dataset + ".png")


if __name__ == '__main__':
    # plotFig_tau(model="CNN", dataset="MNIST", title="", iid=True, mode="tau",
    #         savePath="./FilesAndFigs/p2/",
    #         files=[
    #             open("./FilesAndFigs/p2/tau_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.1_10.txt", 'r'),
    #             open("./FilesAndFigs/p2/tau_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.2.txt", 'r'),
    #             open("./FilesAndFigs/p2/tau_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.5.txt", 'r'),
    #             open("./FilesAndFigs/p2/tau_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_1.0.txt", 'r')
    #         ],
    #         labels=["M=1", "M=2", "M=5", "M=10"])

    # plotFig_tau(model="CNN", dataset="MNIST", title="", iid=True, mode="tau",
    #             savePath="./FilesAndFigs/p3/diff N/",
    #             files=[
    #                 open("./FilesAndFigs/p3/diff N/tau_mnist_20_cnn_100_Gaussian_dp_10.0_epsilon_True_0.05.txt", 'r'),
    #                 open("./FilesAndFigs/p3/diff N/tau_mnist_50_cnn_100_Gaussian_dp_10.0_epsilon_True_0.02.txt", 'r'),
    #                 open("./FilesAndFigs/p3/diff N/tau_mnist_80_cnn_100_Gaussian_dp_10.0_epsilon_True_0.0125.txt", 'r'),
    #                 open("./FilesAndFigs/p3/diff N/tau_mnist_100_cnn_100_Gaussian_dp_10.0_epsilon_True_0.01.txt", 'r')
    #             ],
    #             labels=["M=1", "M=2", "M=5", "M=10"])

    # plotFig_tau(model="CNN", dataset="MNIST", title="", iid=True, mode="tau",
    #             savePath="./FilesAndFigs/p3/diff yita/",
    #             files=[
    #                 open("./FilesAndFigs/p3/diff yita/tau_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.1_10.txt", 'r'),
    #                 open("./FilesAndFigs/p3/diff yita/tau_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.1_50.txt", 'r'),
    #                 open("./FilesAndFigs/p3/diff yita/tau_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.1_100.txt", 'r')
    #             ],
    #             labels=[r"$\gamma=10$", r"$\gamma=50$", r"$\gamma=100$"])

    # plotFig(model="CNN", dataset="MNIST", title="", iid=False, mode="acc",
    #         savePath="./FilesAndFigs/p4/",
    #         files=[
    #             open("./FilesAndFigs/p4/niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.1.txt", 'r'),
    #             open("./FilesAndFigs/p4/niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.3.txt", 'r'),
    #             open("./FilesAndFigs/p4/niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.5.txt", 'r'),
    #             open("./FilesAndFigs/p4/niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.8.txt", 'r'),
    #             open("./FilesAndFigs/p4/niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_1.0.txt", 'r')
    #         ],
    #         labels=[r"$M=0.1\times N$", r"$M=0.3\times N$", r"$M=0.5\times N$", r"$M=0.8\times N$", r"$M=N$"])

    # p5
    # plotFig(model="CNN", dataset="MNIST", title="", iid=True, mode="acc",
    #         savePath="./FilesAndFigs/p5/",
    #         files=[
    #             open("./FilesAndFigs/p5/mnist_iid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_1.0.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_iid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.5.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_iid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.1.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_iid/Semi_mnist_10_cnn_100_no_dp_dp_10.0_epsilon_True_0.2.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_iid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.2.txt", 'r')
    #         ],
    #         labels=["FedAF", "FedAP", "FedAsync", "FedSA", "myFed"])

    # plotFig(model="CNN", dataset="MNIST", title="", iid=False, mode="acc",
    #         savePath="./FilesAndFigs/p5/",
    #         files=[
    #             open("./FilesAndFigs/p5/mnist_niid/Sync_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_1.0.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.5.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.1.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_niid/Semi_mnist_10_cnn_100_no_dp_dp_10.0_epsilon_False_0.8.txt", 'r'),
    #             open("./FilesAndFigs/p5/mnist_niid/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.8.txt", 'r')
    #         ],
    #         labels=["FedAF", "FedAP", "FedAsync", "FedSA", "myFed"])

    # plotFig(model="CNN", dataset="CIFAR", title="", iid=False, mode="acc",
    #         savePath="./FilesAndFigs/p5/",
    #         files=[
    #             open("./FilesAndFigs/p5/cifar_niid/Sync_cifar_10_cnn_100_Gaussian_dp_10.0_epsilon_False_1.0.txt", 'r'),
    #             open("./FilesAndFigs/p5/cifar_niid/Semi_cifar_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.5.txt", 'r'),
    #             open("./FilesAndFigs/p5/cifar_niid/Semi_cifar_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.1.txt", 'r'),
    #             open("./FilesAndFigs/p5/cifar_niid/Semi_cifar_10_cnn_100_no_dp_dp_10.0_epsilon_False_0.8.txt", 'r'),
    #             open("./FilesAndFigs/p5/cifar_niid/Semi_cifar_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.8.txt", 'r')
    #         ],
    #         labels=["FedAF", "FedAP", "FedAsync", "FedSA", "myFed"])

    # plotFig(model="CNN", dataset="MNIST", title="", iid=True, mode="acc",
    #         savePath="./FilesAndFigs/p7/",
    #         files=[
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.2_3.txt", 'r'),
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.2_5.txt", 'r'),
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.2_7.txt", 'r'),
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.2_9.txt", 'r'),
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_True_0.2_11.txt", 'r')
    #         ],
    #         labels=[r"$\tau^0=3$", r"$\tau^0=5$", r"$\tau^0=7$", r"$\tau^0=9$", r"$\tau^0=11$"])

    # plotFig(model="CNN", dataset="MNIST", title="", iid=False, mode="acc",
    #         savePath="./FilesAndFigs/p7/",
    #         files=[
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.2_3.txt", 'r'),
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.2_5.txt", 'r'),
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.2_7.txt", 'r'),
    #             open("./FilesAndFigs/p7/Semi_mnist_10_cnn_100_Gaussian_dp_10.0_epsilon_False_0.2_999.txt", 'r')
    #         ],
    #         labels=[r"$\tau^0=3$", r"$\tau^0=5$", r"$\tau^0=7$", r"$\tau^0=\infty$"])

    a = [13573.63, 3651.75, 1775.46, 1072.52, 746.46, 549.46, 436.08, 371.39, 319.74, 247.77]
    c = [13450.49, 3598.23, 1759.34, 1073.64, 760.91, 547.58, 433.74, 369.25, 317.71, 245.96]
    b = [27963.48, 7373.72, 3389.79, 1981.11, 1298.92, 966.1, 743.12, 648.82, 573.35, 457.47]
    d = [24276.64, 6195.14, 2985.5, 1764.39, 1206.22, 873.09, 677.48, 590.14, 520.96, 416.16]
    t1 = [4629.74, 5072.01, 5809.31, 6370.19, 7261.67, 8046.9, 9369.51, 11218.26, 13660.26, 15452.99]
    t2 = [23803.31, 17188.07, 14001.83, 13177.09, 12185.02, 12105.02, 11720.35, 11521.77, 11275.14, 12067.14]
    # plotFigWithoutFile(title="Predicted finish time of different M", savePath='./FilesAndFigs/p3/predict/',
    #                    filename='Totaltime_iid.png',
    #                    x=[range(1, 11)], y=[a, c],
    #                    xlabel='M', ylabel='time(s)', labels=['MNIST IID', 'CIFAR IID'])
    # plotFigWithoutFile(title="Predicted finish time of different M", savePath='./FilesAndFigs/p3/predict/',
    #                    filename='Totaltime_N-iid.png',
    #                    x=[range(1, 11)], y=[b, d],
    #                    xlabel='M', ylabel='time(s)', labels=['MNIST N-IID', 'CIFAR N-IID'])
    plotFigWithoutFile(title="Predicted finish time of different M", savePath='./FilesAndFigs/p3/predict/',
                       filename='Totaltime_iid(dif from paper).png',
                       x=[range(1, 11)], y=[t1],
                       xlabel='M', ylabel='time(s)', labels=['MNIST IID'])
    plotFigWithoutFile(title="Predicted finish time of different M", savePath='./FilesAndFigs/p3/predict/',
                       filename='Totaltime_N-iid(dif from paper).png',
                       x=[range(1, 11)], y=[t2],
                       xlabel='M', ylabel='time(s)', labels=['MNIST N-IID'])
