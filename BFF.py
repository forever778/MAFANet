import torch
import torch.nn as nn
import torch.nn.functional as F



class BFF1(nn.Module):
    def __init__(self):
        super(BFF1, self).__init__()
        self.left_1 = nn.Sequential(

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )
        self.left_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),




            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)
        )


        self.right_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2,  dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )

        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x_l, x_r):
        lsize = x_l.size()[2:]
        left_1 = self.left_1(x_l)

        left_2 = self.left_2(x_l)

        x_r = F.interpolate(
            x_r, size=lsize, mode='bilinear', align_corners=True)
        right_1 = x_r
        right_2 = self.right_2(x_r)

        left = left_1 * torch.sigmoid(right_1)
        right = left_2 * torch.sigmoid(right_2)
        out = self.conv(left + right)
        return out

class BFF(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(BFF, self).__init__()
        self.left_1 = nn.Sequential(

            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=2, dilation=2, groups=out_chan, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        self.left_2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_chan, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)
        )

        self.right_2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=2,  dilation=2, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()

        )

    def forward(self, x_l, x_r):
        lsize = x_l.size()[2:]
        left_1 = self.left_1(x_l)

        left_2 = self.left_2(x_l)

        x_r = F.interpolate(
            x_r, size=lsize, mode='bilinear', align_corners=True)

        right_2 = self.right_2(x_r)

        left = left_1 * torch.sigmoid(x_r)
        right = left_2 * torch.sigmoid(right_2)
        out = self.conv(left + right)
        return out



