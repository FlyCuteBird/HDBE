""" Image encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch.nn as nn
from torchvision import models
import torch
from models.pie_model import PIENet
from models.uncertainty_module import UncertaintyModuleImage
from utils.tensor_utils import l2_normalize, sample_gaussian_tensors


class EncoderImage(nn.Module):
    def __init__(self, config):
        super(EncoderImage, self).__init__()

        self.embed_dim = config.embed_dim
        self.use_attention = config.img_attention
        self.use_probemb = config.get('img_probemb')

        # Backbone CNN
        self.cnn = getattr(models, config.cnn_type)(pretrained=True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, self.embed_dim)

        self.cnn.fc = nn.Sequential()

        if self.use_attention:
            self.pie_net = PIENet(1, cnn_dim, self.embed_dim, cnn_dim // 2)

        if self.use_probemb:
            self.uncertain_net = UncertaintyModuleImage(cnn_dim, self.embed_dim, cnn_dim // 2)

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = config.img_finetune

        self.n_samples_inference = config.get('n_samples_inference', 0)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
    def image_uniform_sample_points_from_ball(self, x, y, num_samples=7, d_out=1024):
        # 创建一个空的采样结果张量
        samples = []
        for i in range(x.size(0)):
            # 1. 生成随机方向：通过在标准正态分布中采样并归一化
            random_directions = torch.randn(num_samples, d_out).to("cuda") # 随机生成 num_samples 个点
            directions_norm = torch.norm(random_directions, dim=1, keepdim=True).to("cuda")  # 计算每个点的范数
            normalized_directions = random_directions / directions_norm  # 归一化到单位球体

            # 2. 生成随机半径：根据体积分布的原则，使用立方根分布
            u = torch.rand(num_samples, 1).to("cuda")  # 在 [0, 1] 范围内生成均匀随机数
            radii = y[i] * u  # 使用立方根变换生成均匀分布的半径

            # 3. 缩放方向向量：将单位方向向量缩放到目标半径
            scaled_samples = normalized_directions * radii

            # 4. 偏移到球心位置
            shifted_samples = scaled_samples + x[i]

            # 将采样的点添加到结果列表中
            samples.append(shifted_samples)

        # 合并所有的采样点，结果形状为 (4, 5, 1024)
        return torch.stack(samples)

    def image_entropy(self, p):
        """
        计算离散概率分布的信息熵
        """
        return -torch.sum(p * torch.log(p + 1e-20), dim=1, keepdim=True)
    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(pooled)

        output = {}
        out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)

        if self.use_attention:
            out, attn, residual = self.pie_net(out, out_7x7.transpose(1, 2))
            output['attention'] = attn
            # output['residual'] = residual

        # if self.use_probemb:
        #     uncertain_out = self.uncertain_net(pooled, out_7x7.transpose(1, 2))
        #     # logsigma = uncertain_out['logsigma']
        #     # output['logsigma'] = logsigma
        #     output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            # output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
            # c = torch.squeeze(output['uncertainty_attention'])
            c = torch.squeeze(output['attention'])
            # The parameter \lambda is set to 0.01 for COCO and 0.05 fou CUB
            radio = self.image_entropy(c) * 0.01
            output['embedding'] = self.image_uniform_sample_points_from_ball(out, radio, self.n_samples_inference,
                                                                       self.embed_dim)
            output['mean'] = out
            output['radio'] = radio
        else:
            output['embedding'] = out

        return output
