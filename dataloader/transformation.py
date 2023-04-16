
from torchvision.transforms import transforms
import random


def distort(x, num_pixels=1, value=1.0):
    for channel in range(3):
        for _ in range(num_pixels):
            x[channel][int(random.random()*32)][int(random.random()*32)] = value
    return x

def default_cifar10_transforms(to_augment_data):
    if to_augment_data:
        return transforms.Compose([
                            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: distort(x, num_pixels=15)),
                            transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
    else:
        return transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
        
