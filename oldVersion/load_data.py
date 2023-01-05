import torch
import torch.utils.data as Data
from torchvision import datasets, transforms  # 加载计算机视觉有关包
import json


def load_data():
    # 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
    train_dataset = datasets.MNIST(root='~/data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='~/data/', train=False, transform=transforms.ToTensor())

    # 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
    # train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    # test_loader = Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

    return train_dataset, test_dataset


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


def dataLoader(dataset, batch_size, pattern_idx):
    if pattern_idx == 0:
        loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        return loader
    label_clusters = get_label_clusters(pattern_idx)

    loader = []
    batch_data = {}
    batch_target = {}
    for i, (X, label) in enumerate(dataset):
        for j in range(len(label_clusters)):
            if label in label_clusters[j]:
                batch_data.setdefault(j, [])
                batch_data[j].append(X.numpy().tolist())
                batch_target.setdefault(j, [])
                batch_target[j].append(label)
                if len(batch_data[j]) == batch_size:
                    batch_data_tensor = torch.tensor(batch_data[j])
                    batch_target_tensor = torch.tensor(batch_target[j])
                    loader.append([batch_data_tensor, batch_target_tensor])
                    batch_data[j] = []
                    batch_target[j] = []
                break

    return loader


if __name__ == "__main__":
    print("Loading data...")
    (train_dataset, test_dataset) = load_data()

    print("Building data distribution...")
    BATCH_SIZE = 64
    pattern_idx = 1
    print("pattern_idx=", pattern_idx)
    print(type(train_dataset), len(train_dataset), len(test_dataset))
    if 0 < pattern_idx < 1:
        train_dataset_1 = []
        train_dataset_2 = []
        test_dataset_1 = []
        test_dataset_2 = []
        for i, (X, label) in enumerate(train_dataset):
            if i < len(train_dataset) * pattern_idx:
                train_dataset_1.append((X, label))
            else:
                train_dataset_2.append((X, label))
        for i, (X, label) in enumerate(test_dataset):
            if i < len(test_dataset) * pattern_idx:
                test_dataset_1.append((X, label))
            else:
                test_dataset_2.append((X, label))
        print(len(train_dataset_1), len(train_dataset_2))
        train_loader_1 = dataLoader(dataset=train_dataset_1, batch_size=BATCH_SIZE, pattern_idx=0)
        train_loader_2 = dataLoader(dataset=train_dataset_2, batch_size=BATCH_SIZE, pattern_idx=1)
        print("len train_loader_1=", len(train_loader_1))
        print("len train_loader_2=", len(train_loader_2))
        test_loader_1 = dataLoader(dataset=test_dataset_1, batch_size=BATCH_SIZE, pattern_idx=0)
        test_loader_2 = dataLoader(dataset=test_dataset_1, batch_size=BATCH_SIZE, pattern_idx=1)

        torch.save(train_loader_1, "train_loader_" + str(pattern_idx) + "1_" + str(BATCH_SIZE) + ".pt")
        torch.save(test_loader_1, "test_loader_" + str(pattern_idx) + "1_" + str(BATCH_SIZE) + ".pt")
        torch.save(train_loader_2, "train_loader_" + str(pattern_idx) + "2_" + str(BATCH_SIZE) + ".pt")
        torch.save(test_loader_2, "test_loader_" + str(pattern_idx) + "2_" + str(BATCH_SIZE) + ".pt")
    else:
        train_loader = dataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, pattern_idx=pattern_idx)
        test_loader = dataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, pattern_idx=pattern_idx)

        torch.save(train_loader, "train_loader_" + str(pattern_idx) + "_" + str(BATCH_SIZE) + ".pt")
        torch.save(test_loader, "test_loader_" + str(pattern_idx) + "_" + str(BATCH_SIZE) + ".pt")
# torch.save(train_loader, "train_loader_"+str(pattern_idx)+".pt")
# torch.save(test_loader, "test_loader_"+str(pattern_idx)+".pt")
# x2 = torch.load('x.pt')

# f = open("train_loader_"+str(pattern_idx)+".json", 'w')
# jsObj = json.dumps(train_loader)
# f.write(str(train_loader))
# f.close()
