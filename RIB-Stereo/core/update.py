import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, c, x):
        # x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + c[0])
        r = torch.sigmoid(self.convr(hx) + c[1])
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + c[2])
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


    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
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

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * 8
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 96-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

# def interp(x, dest):
#     interp_args = {'mode': 'bilinear', 'align_corners': True}
#     return F.interpolate(x, dest.shape[2:], **interp_args)

def interp(x, dest):
    original_dtype = x.dtype
    x_fp32 = x.float()
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.interpolate(x_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output


class HistoryHighFrequency(nn.Module):
    """Detail candidate from detached historical raw GRU outputs.

    No history list, attention, or channel expansion. Border-aware average
    pooling avoids artificial high frequencies from zero padding on constants.
    This extracts feature-space high frequencies, not guaranteed image edges.
    """
    def __init__(self, hidden_dim=96, strength=0.05):
        super().__init__()
        self.strength = float(strength)
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("history_hf_strength must be finite and in [0, 1]")
        self.detail_conv = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1,
            groups=hidden_dim, bias=False)
        # Identity kernel: initially tanh(high-pass), not random filtering.
        with torch.no_grad():
            self.detail_conv.weight.zero_()
            self.detail_conv.weight[:, 0, 1, 1] = 1.0

    def forward(self, raw_net, history_mean=None):
        if history_mean is None or self.strength == 0.0:
            return raw_net
        # Stop gradient only through the cache. DO NOT wrap the detail
        # convolution in no_grad: its weights must remain trainable.
        memory = history_mean.detach().to(dtype=raw_net.dtype)
        low = F.avg_pool2d(memory, 3, stride=1, padding=1,
                           count_include_pad=False)
        high = memory - low
        detail = torch.tanh(self.detail_conv(high))
        return raw_net + self.strength * detail


@torch.no_grad()
def update_history_mean(history_mean, count, raw_net):
    """Append the uncompensated GRU output exactly once (out of place).

    The next iteration sees mean(u_1,...,u_t). Initial h_0 and compensated
    states are not directly added. Raw outputs still depend indirectly on
    earlier compensation through the ordinary recurrent input.
    """
    state = raw_net.detach()
    if history_mean is None:
        if count != 0:
            raise ValueError("Empty history requires count=0")
        return state.clone(), 1
    if count < 1:
        raise ValueError("Nonempty history requires positive count")
    count += 1
    state = state.to(history_mean.dtype)
    return history_mean + (state - history_mean) / count, count

class HistoryHighFrequency_mutil(nn.Module):
    def __init__(self, hidden_dim=96, strength=0.05):
        super().__init__()
        self.strength = float(strength)
        # 深度可分离，参数量 2400 + 4704 = 7104
        self.low5 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2,
                              groups=hidden_dim, bias=False)
        self.low7 = nn.Conv2d(hidden_dim, hidden_dim, 7, padding=3,
                              groups=hidden_dim, bias=False)
        with torch.no_grad():
            self.low5.weight.fill_(1.0 / 25.0)   # 初始 = 5x5 均值滤波
            self.low7.weight.fill_(1.0 / 49.0)   # 初始 = 7x7 均值滤波
        self.detail_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                                     groups=hidden_dim, bias=False)
        with torch.no_grad():
            self.detail_conv.weight.zero_()
            self.detail_conv.weight[:, 0, 1, 1] = 1.0

    def forward(self, raw_net, history_mean=None):
        if history_mean is None or self.strength == 0.0:
            return raw_net
        memory = history_mean.detach().to(dtype=raw_net.dtype)
        memory = F.pad(memory, (3, 3, 3, 3), mode='reflect')  # 防零填充伪影
        low = 0.5 * (self.low5(memory) + self.low7(memory))  # 多尺度低通
        high = raw_net * 0 + (history_mean.detach().to(raw_net.dtype) - low)  # 用未 pad 的 memory
        detail = torch.tanh(self.detail_conv(high))
        return raw_net + self.strength * detail


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = hidden_dim

        input_dim = 96
        self.gru = ConvGRU(hidden_dim, input_dim)
        self.history_hf = HistoryHighFrequency(
            hidden_dim, getattr(args, "history_hf_strength", 0.05))
        self.disp_head = DispHead(hidden_dim, hidden_dim=128, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None,
                history_mean=None, return_raw=False):

        motion_features = self.encoder(disp, corr)
        raw_net = self.gru(net, inp, motion_features)
        net = self.history_hf(raw_net, history_mean)

        delta_disp = self.disp_head(net)
        mask_feat_4 = self.mask_feat_4(net)
        if return_raw:
            return net, mask_feat_4, delta_disp, raw_net
        return net, mask_feat_4, delta_disp
