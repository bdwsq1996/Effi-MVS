import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *

class DepthHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x_d, act_fn=torch.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        return act_fn(out)

class ConvGRU_new(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU_new, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class ProjectionInputDepth(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convd = nn.Conv2d(64+hidden_dim, out_chs - 1, 3, padding=1)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, depth, cost):
        # print(cost.size())
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        dfm = F.relu(self.convd1(depth))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cor, dfm], dim=1)

        out_d = F.relu(self.convd(cor_dfm))
        if self.training and self.dropout is not None:
            out_d = self.dropout(out_d)
        return torch.cat([out_d, depth], dim=1)

class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask

class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=256, ratio=8, context_dim=64 ,UpMask=False):
        super(BasicUpdateBlockDepth, self).__init__()

        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.depth_head = DepthHead(hidden_dim, hidden_dim=hidden_dim, scale=False)
        self.UpMask = UpMask
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, net, depth_cost_func, inv_depth, context, seq_len=4, scale_inv_depth=None):
        inv_depth_list = [] 
        mask_list = []
        for i in range(seq_len):

            # TODO detach()
            inv_depth = inv_depth.detach()

            input_features = self.encoder(inv_depth, depth_cost_func(scale_inv_depth(inv_depth)[1]))

            inp_i = torch.cat([context, input_features], dim=1)

            net = self.depth_gru(net, inp_i)

            delta_inv_depth = self.depth_head(net)

            inv_depth = inv_depth + delta_inv_depth
            inv_depth_list.append(inv_depth)
            if self.UpMask and i == seq_len - 1 :
                mask = .25 * self.mask(net)
                mask_list.append(mask)
            else:
                mask_list.append(inv_depth)
        return net, mask_list, inv_depth_list
