"""
Code referenced from TAC-ICML-2024:
@inproceedings{
  li2024image,
  title={Image Clustering with External Guidance},
  author={Li, Yunfan and Hu, Peng and Peng, Dezhong and Lv, Jiancheng and Fan, Jianping and Peng, Xi},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=JSYN891WnB}
}
Github Link: https://github.com/XLearning-SCU/2024-ICML-TAC/tree/main
"""

# import faiss
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
import copy
import torch
from models import CLIPModel

BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(dataset="CIFAR-10"):
    if (
        dataset == "CIFAR-10"
        or dataset == "CIFAR-20"
        or dataset == "STL-10"
        or dataset == "DTD"
        or dataset == "UCF101"
    ):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif (
        dataset == "ImageNet-Dogs" or dataset == "ImageNet-10" or dataset == "ImageNet"
    ):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        raise NotImplementedError
    return transforms


def get_dataloader(dataset="CIFAR-10", batch_size=4096):
    transforms = get_transforms(dataset)
    if dataset == "CIFAR-10":
        data_train = CIFAR10(
            root="./data", train=True, download=True, transform=transforms
        )
        data_test = CIFAR10(
            root="./data", train=False, download=True, transform=transforms
        )
    elif dataset == "CIFAR-20":
        data_train = CIFAR100(
            root="./data", train=True, download=True, transform=transforms
        )
        data_test = CIFAR100(
            root="./data", train=False, download=True, transform=transforms
        )
    elif dataset == "STL-10":
        data_train = STL10(
            root="./data", split="train", download=True, transform=transforms
        )
        data_test = STL10(
            root="./data", split="test", download=True, transform=transforms
        )
    elif dataset == "ImageNet-10":
        data_train = ImageFolder("./data/ImageNet-10/train", transform=transforms)
        data_test = ImageFolder("./data/ImageNet-10/val", transform=transforms)
    elif dataset == "ImageNet-Dogs":
        data_train = ImageFolder("./data/ImageNet-Dogs/train", transform=transforms)
        data_test = ImageFolder("./data/ImageNet-Dogs/val", transform=transforms)
    elif dataset == "DTD":
        data_train = ImageFolder("./data/DTD/trainval", transform=transforms)
        data_test = ImageFolder("./data/DTD/test", transform=transforms)
    elif dataset == "UCF101":
        data_train = ImageFolder("./data/UCF101/train", transform=transforms)
        data_test = ImageFolder("./data/UCF101/val", transform=transforms)
    elif dataset == "ImageNet":
        # data_test = ImageFolder("./data/ImageNet/val", transform=transforms)
        data_test = ImageFolder("/home/elef/Data/ImageNet1K", transform=transforms)
    else:
        raise NotImplementedError

    if dataset != "ImageNet":

        dataloader_train = DataLoader(
            data_train, batch_size=batch_size, shuffle=False, drop_last=False
        )
        dataloader_test = DataLoader(
            data_test, batch_size=batch_size, shuffle=False, drop_last=False
        )

        return dataloader_train, dataloader_test

    else:
        dataloader_test = DataLoader(
            data_test, batch_size=batch_size, shuffle=False, drop_last=False
        )

        return dataloader_test



dataset = "ImageNet" #　["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
if dataset != 'ImageNet':
    dataloader_train, dataloader_test = get_dataloader(
        dataset=dataset, batch_size=1024
    )
else:
    dataloader_test = get_dataloader(
        dataset=dataset, batch_size=1024
    )

model = CLIPModel(model_name="ViT-B/32").cuda()
model.eval()

features = []
labels = []
print("Inferring image features and labels...")
if dataset != 'ImageNet':
    for iteration, (x, y) in enumerate(dataloader_train):
        x = x.cuda()
        with torch.no_grad():
            feature = model.encode_image(x)
        features.append(feature.cpu().numpy())
        labels.append(y.numpy())
        if iteration % 10 == 0:
            print(f"[Iter {iteration}/{len(dataloader_train)}]")
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print("Feature shape:", features.shape, "Label shape:", labels.shape)

features_test = []
labels_test = []
print("Inferring test image features and labels...")
for iteration, (x, y) in enumerate(dataloader_test):
    x = x.cuda()
    with torch.no_grad():
        feature = model.encode_image(x)
    features_test.append(feature.cpu().numpy())
    labels_test.append(y.numpy())
    if iteration % 10 == 0:
        print(f"[Iter {iteration}/{len(dataloader_test)}]")
features_test = np.concatenate(features_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)
print("Feature shape:", features_test.shape, "Label shape:", labels_test.shape)

if dataset == "CIFAR-20":
    coarse_label = [
        [72, 4, 95, 30, 55],
        [73, 32, 67, 91, 1],
        [92, 70, 82, 54, 62],
        [16, 61, 9, 10, 28],
        [51, 0, 53, 57, 83],
        [40, 39, 22, 87, 86],
        [20, 25, 94, 84, 5],
        [14, 24, 6, 7, 18],
        [43, 97, 42, 3, 88],
        [37, 17, 76, 12, 68],
        [49, 33, 71, 23, 60],
        [15, 21, 19, 31, 38],
        [75, 63, 66, 64, 34],
        [77, 26, 45, 99, 79],
        [11, 2, 35, 46, 98],
        [29, 93, 27, 78, 44],
        [65, 50, 74, 36, 80],
        [56, 52, 47, 59, 96],
        [8, 58, 90, 13, 48],
        [81, 69, 41, 89, 85],
    ]
    labels_copy = copy.deepcopy(labels)
    labels_test_copy = copy.deepcopy(labels_test)
    for i in range(20):
        for j in coarse_label[i]:
            labels[labels_copy == j] = i
            labels_test[labels_test_copy == j] = i

if dataset != 'ImageNet':
    np.save("./data/" + dataset + "_image_embedding_train.npy", features)
    np.save("./data/" + dataset + "_image_embedding_test.npy", features_test)
    np.savetxt("./data/" + dataset + "_labels_train.txt", labels)
    np.savetxt("./data/" + dataset + "_labels_test.txt", labels_test)

else:
    np.save("./data/" + dataset + "_image_embedding_test.npy", features_test)



t = 10