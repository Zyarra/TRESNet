import PIL
import torch
import torchvision
from hyperparameters import IMAGE_SIZE


def data_preparation(data_dir, batch_size_train, batch_size_test):
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=PIL.Image.BILINEAR),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=PIL.Image.BILINEAR),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
    ds = torchvision.datasets.ImageFolder(data_dir)

    val_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=PIL.Image.BILINEAR),
         torchvision.transforms.ToTensor()])

    train_ds, test_ds = torch.utils.data.random_split(ds, [30715, 7000])
    test_ds, val_ds = torch.utils.data.random_split(test_ds, [4000, 3000])
    train_ds.dataset.transform = train_transform
    test_ds.dataset.transform = test_transform
    val_ds.dataset.transform = val_transform

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train,
                                               shuffle=True, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test,
                                              shuffle=True, pin_memory=True, drop_last=True)

    printable = f'\nNumber of images: Train:[{len(train_ds)}], Test:[{len(test_ds)}]'  # Validation:[{len(val_ds)}]'
    print(printable)
    print('-' * len(printable))
    return train_loader, test_loader
