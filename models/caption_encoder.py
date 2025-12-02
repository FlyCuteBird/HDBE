""" Caption encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_pretrained_bert import BertModel
import torchtext
import numpy as np
from models.pie_model import PIENet
from models.uncertainty_module import UncertaintyModuleText
from utils.tensor_utils import l2_normalize, sample_gaussian_tensors


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask

class WordEmbeddings(nn.Module):
    def __init__(self):
        super(WordEmbeddings, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)[0]
        x = x[-1:][0]
        # x= torch.Tensor(x)
        # x = torch.stack(x, dim=1)
        # x = torch.sum(x, dim=1)
        return x

class EncoderText(nn.Module):
    def __init__(self, word2idx, opt):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_dim = \
            opt.wemb_type, opt.word_dim, opt.embed_dim

        self.embed_dim = embed_dim
        self.use_attention = opt.txt_attention
        self.use_probemb = opt.get('txt_probemb')

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = opt.txt_finetune

        # # # bert embedding
        # self.embed = WordEmbeddings()

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        if self.use_attention:
            self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)

        self.uncertain_net = UncertaintyModuleText(word_dim, embed_dim, word_dim // 2)
        self.init_weights(wemb_type, word2idx, word_dim, opt.cache_dir)

        self.n_samples_inference = opt.get('n_samples_inference', 0)

    def init_weights(self, wemb_type, word2idx, word_dim, cache_dir):
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))
    def uniform_sample_points_from_ball(self, x, y, num_samples=7, d_out=1024):
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

    def entropy(self, p):
        """
        计算离散概率分布的信息熵
        """
        return -torch.sum(p * torch.log(p + 1e-20), dim=1, keepdim=True)

    def forward(self, x, lengths):
        # Embed word ids to vectors
        wemb_out = self.embed(x)
        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)
        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        I = I.to("cuda")
        out = torch.gather(padded[0], 1, I).squeeze(1)

        output = {}

        if self.use_attention:
            pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True).to("cuda") # to("cuda") Qin
            out, attn, residual = self.pie_net(out, wemb_out, pad_mask)
            output['attention'] = attn
            # output['residual'] = residual

        # if self.use_probemb:
        #     if not self.use_attention:
        #         pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True).to("cuda") #to("cuda") Qin
        #     uncertain_out = self.uncertain_net(wemb_out, pad_mask, lengths)
        #     # logsigma = uncertain_out['logsigma']
        #     # output['logsigma'] = logsigma
        #     output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            # output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
            # c = torch.squeeze(output['uncertainty_attention'])
            c = torch.squeeze(output['attention'])
            # scale_radio = 1.0 / torch.unsqueeze(torch.log2(lengths), dim=1).to("cuda")
            # radio = torch.mul(self.entropy(c),scale_radio)
            # The parameter \lambda is set to 0.01 for COCO and 0.05 fou CUB
            radio = self.entropy(c) * 0.01
            output['embedding'] = self.uniform_sample_points_from_ball(out, radio, self.n_samples_inference, self.embed_dim)
            output['mean'] = out
            output['radio'] = radio
        else:
            output['embedding'] = out

        return output
