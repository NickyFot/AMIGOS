from torchvision import transforms
from amigos_dataset import AMIGOS


if __name__ == '__main__':
    amigos = AMIGOS(
        root_path='Frames',
        annotation_path='amigos.json',
        spatial_transform=transforms.ToTensor(),
        feature_type='RGB'
    )
    print(len(amigos))
    sample = amigos[0]
    print(sample[0].shape, sample[1].shape)
