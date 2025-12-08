import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, img_size, dim, num_heads=8, context_dim=None):
        super(Attention, self).__init__()
        # [修改] PosEmbedding 保持为 dim 维度 (与 Content 一致)
        self.pos_embbedding = nn.Parameter(torch.randn(1, dim, img_size, img_size))

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 基础投影层
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.mat = torch.matmul
        self.proj = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim * 2, dim, (1, 1))

        # [新增] 适配器：如果组件特征维度 (context_dim) 与 Content (dim) 不一致，则投影
        self.context_adapter = None
        if context_dim is not None and context_dim != dim:
            self.context_adapter = nn.Linear(context_dim, dim)

    def forward(self, content_feat, component_list):
        B, C, H, W = content_feat.shape

        # 1. 准备 PosEmbedding
        pos_emb = F.interpolate(self.pos_embbedding, size=(H, W), mode='bilinear', align_corners=False)
        pos_emb = pos_emb.flatten(2).transpose(1, 2) # [1, HW, dim]

        # 2. 处理 Content (Query)
        content_token = content_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        content_token = content_token + pos_emb # [B, HW, dim]

        # 3. 处理 Components (Key/Value)
        comp_token = []
        for i in range(len(component_list)):
            comp = component_list[i]

            # 插值对齐 H, W
            if comp.shape[2:] != (H, W):
                comp = F.interpolate(comp, size=(H, W), mode='bilinear', align_corners=False)

            # [修改] 使用组件自身的通道数进行 reshape
            C_comp = comp.shape[1]
            token = comp.permute(0, 2, 3, 1).reshape(B, H * W, C_comp) # [B, HW, 256]

            # [新增] 维度适配: 256 -> 128
            if self.context_adapter is not None:
                token = self.context_adapter(token)
            
            # 现在 token 是 [B, HW, dim(128)]，可以安全相加了
            token = token + pos_emb
            comp_token.append(token)

        # 4. Attention 计算
        query = self.q(content_token).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        keys = []
        values = []
        for i in range(len(comp_token)):
            kv = self.kv(comp_token[i]).reshape(B, H * W, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            key, value = kv[0], kv[1]
            keys.append(key)
            values.append(value)

        result = 0
        for i in range(len(keys)):
            attn = (self.mat(query, keys[i].transpose(-2, -1))) * self.scale
            attn = attn.float().softmax(dim=-1).type_as(query)
            res = self.mat(attn, values[i]).transpose(1, 2).reshape(B, H * W, C)
            result = result + res

        s = result + content_token
        s = self.proj(s)

        # 还原维度
        s = s.transpose(1, 2).reshape(B, C, H, W)

        feat_c_s = torch.cat((s, content_feat), dim=1)
        feat_c_s = self.conv(feat_c_s)

        return feat_c_s
