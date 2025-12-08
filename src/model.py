import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

class FontDiffuserModel(ModelMixin, ConfigMixin):
    """
    Forward function for FontDiffuser with content encoder,
    style encoder and unet.
    """

    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        content_encoder_downsample_size,
        component_list=None,
    ):
        # 1. Style Encoder
        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        # 2. Content Encoder (Content Image)
        # 接收三个返回值：内容特征、残差、组件特征列表
        content_img_feature, content_residual_features, component_feats = self.content_encoder(
            content_images, 
            component_list=component_list
        )
        content_residual_features.append(content_img_feature)
        
        # 3. Content Encoder (Style Reference Image)
        # 风格图不需要组件特征，用 _ 忽略第三个返回值
        style_content_feature, style_content_res_features, _ = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            style_residual_features
        ]
        
        # 4. UNet
        # 将提取好的 component_feats 传递给 UNet
        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
            component_list=component_feats
        )
        noise_pred = out[0]
        offset_out_sum = out[1]
        
        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """
    DPM Forward function for FontDiffuser with content encoder,
    style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
        component_list=None
    ):
        content_images = cond[0]
        style_images = cond[1]

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # Get content feature
        # 注意：这里也需要适配 ContentEncoder 的 3 个返回值
        content_img_feature, content_residual_features, component_feats = self.content_encoder(
            content_images,
            component_list=component_list
        )
        content_residual_features.append(content_img_feature)
        
        # Get the content feature from reference image
        style_content_feature, style_content_res_features, _ = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            style_residual_features
        ]
        
        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
            component_list=component_feats # 传递组件特征
        )
        noise_pred = out[0]
        
        return noise_pred

