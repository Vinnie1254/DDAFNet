import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x



class ECAM(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ECAM, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())
    def forward(self, x):
        y = self.attention(x)
        return x * y



class LCF(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(LCF, self).__init__()
        self.lcf = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ECAM(num_feat, squeeze_factor)
            )
    def forward(self, x):
        return self.lcf(x)



class FEB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FEB, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.FDEB = FDEB(num_fea=out_channel)

        self.SDEB = SDEB(out_channel)

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x1 = self.FDEB(x)
        x2 = self.SDEB(x)

        weight_sum = self.alpha1 + self.alpha2 + 1e-6
        w1 = self.alpha1 / weight_sum
        w2 = self.alpha2 / weight_sum


        x_out = w1 * x1 + w2 * x2

        return self.main(x_out) + x



class SDEB(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()


        self.SDEB_K_5 = SDEB_K(k, kernel=5)
        self.SDEB_K_7 = SDEB_K(k, kernel=7)

        self.conv = nn.Conv2d(k, k, 1)
    def forward(self, x):

        SDEB_K_5_out = self.SDEB_K_5(x)
        SDEB_K_7_out = self.SDEB_K_7(x)

        out = SDEB_K_7_out + SDEB_K_5_out

        return self.conv(out)


class SDEB_K(nn.Module):
    def __init__(self, k, kernel=7) -> None:
        super().__init__()

        self.channel = k


        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))


        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))


        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)

        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)


        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))


        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):

        hori_l = self.hori_pool(self.pad_hori(x))

        hori_h = x - hori_l


        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h


        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l


        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h


        return x * self.beta + vert_out * self.gamma

class CALayer(nn.Module):
    def __init__(self, num_fea):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 8, num_fea, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fea):
        return self.conv_du(fea)

class AFB(nn.Module):
    def __init__(self, num_fea):
        super(AFB, self).__init__()
        self.CA1=CALayer(num_fea)
        self.CA2=CALayer(num_fea)
        self.fuse=nn.Conv2d(num_fea*2,num_fea,1)
    def forward(self,x1,x2):
        x1=self.CA1(x1)*x1
        x2=self.CA2(x2)*x2

        min_h = min(x1.shape[2], x2.shape[2])
        min_w = min(x1.shape[3], x2.shape[3])
        x1 = x1[:, :, :min_h, :min_w]
        x2 = x2[:, :, :min_h, :min_w]
        return self.fuse(torch.cat((x1, x2), dim=1))





class FDEB(nn.Module):
    def __init__(self, num_fea):
        super(FDEB, self).__init__()
        self.channel1=num_fea//2
        self.channel2=num_fea-self.channel1
        self.convblock = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        )
        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)

    def forward(self, x):
        x1,x2=torch.split(x,[self.channel1,self.channel2],dim=1)

        x1 = self.convblock(x1)

        A = self.A_att_conv(x1)
        S1 = torch.cat((x2, A*x1),dim=1)

        B = self.B_att_conv(x2)
        S2 = torch.cat((x1, B*x2),dim=1)

        c=torch.cat((self.fuse1(S1),self.fuse2(S2)),dim=1)
        out=self.fuse(c)
        return out