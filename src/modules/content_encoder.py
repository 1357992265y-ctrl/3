import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Parameter as P

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)
# ================= [修改] 使用相对导入 =================
# 这里的 .Muti_attention 表示在当前目录 (src/modules/) 下查找文件
try:
    from .Muti_attention import Attention
except ImportError:
    # 备用：如果上面的失败，尝试从 src.modules 导入（适用于不同启动方式）
    import sys
    from src.modules.Muti_attention import Attention

# 同理，CCAM_Attention 也统一用相对导入
try:
    from .Muti_attention import Attention as CCAM_Attention
except ImportError:
    from src.modules.Muti_attention import Attention as CCAM_Attention
# ======================================================

def proj(x, y):
   return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


class LinearBlock(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        norm='none', 
        act='relu', 
        use_sn=False
    ):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        # 建立普通线性层： out=Wx+b
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    """多层感知机：由多个LinearBlock组成"""
    def __init__(
        self, 
        nf_in, 
        nf_out, 
        nf_mlp, 
        num_blocks, 
        norm, 
        act, 
        use_sn =False
    ):
        super(MLP,self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm = norm, act = act, use_sn = use_sn))
        for _ in range((num_blocks - 2)):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act ='none', use_sn = use_sn))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class SN(object):
    def __init__(
        self, 
        num_svs, 
        num_itrs, 
        num_outputs, 
        transpose=False, 
        eps=1e-12
    ):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]

class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward_wo_sn(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class Attention(nn.Module):
    def __init__(
        self, 
        ch, 
        which_conv=SNConv2d, 
        name='attention'
    ):
        super(Attention, self).__init__()
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels

        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers创建两层卷积：
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                            kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d,which_bn= nn.BatchNorm2d, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv,self.which_bn =which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    
    def forward(self, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GBlock2(nn.Module):
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, activation=None,
                upsample=None, skip_connection = True):
        super(GBlock2, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv = which_conv
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                            kernel_size=1, padding=0)

        # upsample layers
        self.upsample = upsample
        self.skip_connection = skip_connection

    def extract_features(self, x):
        """
        辅助函数：专门用于将【部件图像】转换为【部件特征 fc_i】
        这对应文档中的 "处理每个组件特征 fc_i"
        """
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        return h

        # ================= 重点修改了这里 =================
    def forward(self, x, component_list=None):  # <--- 新增 component_list 参数
            """
            Args:
                x: 内容图像 (Content Image)
                component_list: 部件图像列表 (List of Component Images) [B, 3, H, W]
            """
            h = x
            residual_features = []
            residual_features.append(h)

            # 1. 提取内容特征 fs (文档中的第一步)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
                if index in self.save_featrues[:-1]:
                    residual_features.append(h)

            # 2. 如果启用了 CCAM 并且传入了部件图像
            if self.use_ccam and component_list is not None:
                # component_list 可能是列表，也可能是 Tensor [B, K, C, H, W]
                # 我们统一处理成列表形式的特征

                comp_feats = []  # 存放文档中提到的 {fc_i}

                # 如果是 Tensor (Batching 处理过的)，需要拆解
                if isinstance(component_list, torch.Tensor):
                    # shape: [B, K, 3, 128, 128]
                    B, K, C, H, W = component_list.shape
                    # 为了加速，我们可以把 K 和 B 合并一起通过 Encoder
                    # [B*K, 3, 128, 128]
                    flat_imgs = component_list.view(B * K, C, H, W)
                    flat_feats = self.extract_features(flat_imgs)  # 提取特征

                    # 还原维度 [B, K, C_out, H_f, W_f]
                    _, C_out, H_f, W_f = flat_feats.shape
                    flat_feats = flat_feats.view(B, K, C_out, H_f, W_f)

                    # 转回 list 传给 CCAM
                    for k in range(K):
                        comp_feats.append(flat_feats[:, k, ...])

                # 如果本身就是 List (未 Batching 或特殊情况)
                elif isinstance(component_list, list):
                    for comp_img in component_list:
                        feat = self.extract_features(comp_img)
                        comp_feats.append(feat)

                # 3. CCAM 融合 (文档中的第二、三步)
                if len(comp_feats) > 0:
                    # h 是内容特征 fs
                    # comp_feats 是组件特征集合 {fc_i}
                    h = self.ccam_module(h, comp_feats)  # 调用 Muti_attention

            return h, residual_features

def content_encoder_arch(ch =64,out_channel_multiplier = 1, input_nc = 3):
    arch = {}
    n=2
    arch[80] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2]],
                                'out_channels' : [item * ch for item in [1,2,4]],
                                'resolution': [40,20,10]}
    arch[96] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2]],
                                'out_channels' : [item * ch for item in [1,2,4]],
                                'resolution': [48,24,12]}
                                
    arch[128] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                                'resolution': [64,32,16,8,4]}
    
    arch[256] = {'in_channels':[input_nc]+[ch*item for item in [1,2,4,8,8]],
                                'out_channels':[item*ch for item in [1,2,4,8,8,16]],
                                'resolution': [128,64,32,16,8,4]}
    return arch

class ContentEncoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, G_ch=64, G_wide=True, resolution=128, 
                 G_kernel_size=3, G_attn='64_32_16_8', n_classes=1000, 
                 num_G_SVs=1, num_G_SV_itrs=1, G_activation=nn.ReLU(inplace=False), 
                 SN_eps=1e-12, output_dim=1,  G_fp16=False, 
                 G_init='N02',  G_param='SN', nf_mlp = 512, nEmbedding = 256,
                 input_nc = 3,output_nc = 3,use_ccam=True):
        super(ContentEncoder, self).__init__()

        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16
        self.use_ccam = use_ccam  # 保存参数
        if self.resolution == 96:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 80:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 128:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 256:
            self.save_featrues = [0,1,2,3,4,5]
        
        self.out_channel_nultipiler = 1
        self.arch = content_encoder_arch(self.ch,
                                         self.out_channel_nultipiler,
                                         input_nc)[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):

            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.G_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=nn.AvgPool2d(2))]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        if self.use_ccam:
            # 获取最后一层的通道数和分辨率
            last_dim = self.arch['out_channels'][-1]
            last_res = self.arch['resolution'][-1]

            # 实例化 CCAM Attention
            # 注意：num_heads 可以作为参数传入，这里默认设为 8
            # img_size 必须与特征图大小一致，因为原代码中有固定位置编码
            self.ccam_module = CCAM_Attention(img_size=last_res, dim=last_dim, num_heads=8)
        # ============================
        self.init_weights()


    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def extract_features(self, x):
        """辅助函数：通过 block 提取特征"""
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        return h

        # [修改] 添加 component_list=None 参数
        # [修改] 添加 component_list=None 参数
    # [修改] 替换整个 forward 函数
    def forward(self, x, component_list=None):
        # 1. 处理 5D 输入 (K-shot Style Images)
        is_k_shot = False
        if x.dim() == 5:
            is_k_shot = True
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)

        h = x
        residual_features = []
        residual_features.append(h)
        
        # 2. 正常卷积提取 Content 特征
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)            
            if index in self.save_featrues[:-1]:
                residual_features.append(h)
        
        # 3. 提取组件特征 (CCAM)
        final_comp_feats_list = []  # [新增] 用于存储组件特征
        
        if self.use_ccam and component_list is not None:
            # A. 输入是 Tensor [B, K, C, H, W]
            if isinstance(component_list, torch.Tensor) and component_list.dim() == 5:
                B_c, K_c, C_c, H_c, W_c = component_list.shape
                # 展平 -> 提取 -> 还原 -> 转置
                flat_imgs = component_list.view(B_c * K_c, C_c, H_c, W_c)
                flat_feats = self.extract_features(flat_imgs)  # [B*K, 256, 12, 12]
                _, C_f, H_f, W_f = flat_feats.shape
                
                reshaped_feats = flat_feats.view(B_c, K_c, C_f, H_f, W_f)
                transposed_feats = reshaped_feats.permute(1, 0, 2, 3, 4) # [K, B, C, H, W]
                
                for k in range(K_c):
                    final_comp_feats_list.append(transposed_feats[k])
            
            # B. 输入是 List
            elif isinstance(component_list, list) and len(component_list) > 0:
                for comp_img in component_list:
                    feat = self.extract_features(comp_img)
                    final_comp_feats_list.append(feat)

            # 执行 CCAM 融合 (仅针对非 K-shot 的 Content Image)
            if len(final_comp_feats_list) > 0 and not is_k_shot:
                h = self.ccam_module(h, final_comp_feats_list)

        # 4. 还原 5D 输入 (K-shot Style Images)
        if is_k_shot:
            _, C_out, H_out, W_out = h.shape
            h = h.view(B, K, C_out, H_out, W_out).mean(dim=1)
            
            new_residuals = []
            for res in residual_features:
                _, C_r, H_r, W_r = res.shape
                res = res.view(B, K, C_r, H_r, W_r).mean(dim=1)
                new_residuals.append(res)
            residual_features = new_residuals

        # [修改] 返回三个值：内容特征，残差，组件特征列表
        return h, residual_features, final_comp_feats_list
