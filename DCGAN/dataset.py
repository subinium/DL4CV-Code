import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data


def data(root, image_size=64, batch_size=128, num_workers=2):
    dataset = dset.ImageFolder(root=root,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)

    return dataloader
