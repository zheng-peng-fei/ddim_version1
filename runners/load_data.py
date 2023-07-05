import torch
import torchvision.datasets as dset
import torchvision.transforms as trn

cifar10_path = 'data/cifarpy'
def load_CIFAR(batch_size):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
    #                                trn.ToTensor(), trn.Normalize(mean, std)])
    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    print('loading CIFAR-10')
    train_data = dset.CIFAR10(
        cifar10_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(
        cifar10_path, train=False, transform=test_transform, download=True)

    train = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True, pin_memory=True)
    return train