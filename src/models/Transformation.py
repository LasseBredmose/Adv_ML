import random
from imgaug import augmenters as iaa
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

def ChooseTrans(translist):
    rand_trans=random.choice(translist)
    if rand_trans == 'Blank':
        trans =  RandomRotation(0)
    
    if rand_trans == 'FlipV':
        trans = RandomVerticalFlip(1)
    
    if rand_trans == 'FlipH':
        trans = RandomHorizontalFlip(1)
    
    if rand_trans == 'Rotate90':
        trans =  RandomRotation(90)

    if rand_trans == 'Rotate180':
        trans =  RandomRotation(180)

    if rand_trans == 'Rotate270':
        trans =  RandomRotation(270)

    return trans