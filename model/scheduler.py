# Code taken from https://github.com/Kevin-thu/DiffMorpher
# Perceptual uniform distance sampling after finded the path
import bisect
import torch
import torch.nn.functional as F
import lpips

class Scheduler:
    def __init__(self, device='cuda:0'):
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    # perceptual_loss = lpips.LPIPS()

    def distance(self, img_a, img_b):
        return self.perceptual_loss(img_a, img_b).item()
    
    def from_imgs(self, imgs):
        self.__num_values = len(imgs)
        self.__values = [0]
        for i in range(self.__num_values - 1):
            dis = self.distance(imgs[i], imgs[i + 1])
            self.__values.append(dis)
            self.__values[i + 1] += self.__values[i]
        for i in range(self.__num_values):
            self.__values[i] /= self.__values[-1]

    def save(self, filename):
        torch.save(torch.tensor(self.__values), filename)

    def load(self, filename):
        self.__values = torch.load(filename).tolist()
        self.__num_values = len(self.__values)

    def get_x(self, y):
        assert y >= 0 and y <= 1
        id = bisect.bisect_left(self.__values, y)
        id -= 1
        if id < 0:
            id = 0
        yl = self.__values[id]
        yr = self.__values[id + 1]
        xl = id * (1 / (self.__num_values - 1))
        xr = (id + 1) * (1 / (self.__num_values - 1))
        x = (y - yl) / (yr - yl) * (xr - xl) + xl
        return x

    def get_list(self, len=None):
        if len is None:
            len = self.__num_values

        ys = torch.linspace(0, 1, len)
        res = [self.get_x(y) for y in ys]
        return res


