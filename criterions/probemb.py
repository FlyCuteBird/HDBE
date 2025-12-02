"""Batch-wise efficient probabilistic embedding loss for cross-modal retrieval

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def batchwise_cdist(samples1, samples2, eps=1e-6):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)

    The following broadcasting operation will be computed:
    (N x 1 x K x D) - (N x K x 1 x D) = (N x K x K x D)

    Parameters
    ----------
    samples1: torch.Tensor (shape: N x K x D)
    samples2: torch.Tensor (shape: N x K x D)

    Returns
    -------
    batchwise distance: N x K ** 2
    """
    if len(samples1.size()) != 3 or len(samples2.size()) != 3:
        raise RuntimeError('expected: 3-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')

    samples1 = samples1.unsqueeze(1)
    samples2 = samples2.unsqueeze(2)
    return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = True # Hardest strategy
        self.Rank_Loss = False  # Dynamic TopK strategy

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        # torch.eye()函数生成对角线为1.其余全为0.的二维float型tensor；>.5是为了生成对角线为1，其余为0的整数tensor
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.Rank_Loss:
            topK = int((cost_s > 0.).sum() / (cost_s.size(0) + 0.00001) + 1)
            cost_s, index1 = torch.sort(cost_s, descending=True, dim=-1)
            cost_im, index2 = torch.sort(cost_im, descending=True, dim=0)

            return cost_s[:, 0:topK].sum() + cost_im[0:topK, :].sum()

        elif self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

            return cost_s.sum() + cost_im.sum()

        else:
            return cost_s.sum() + cost_im.sum()
        # if self.max_violation:
        #     cost_s = cost_s.max(1)[0]
        #     cost_im = cost_im.max(0)[0]
        # return cost_s.sum() + cost_im.sum()


# def Sphere_dist1(
#     X: torch.Tensor,         # shape (B, N, D): samples in each A_i
#     centers_B: torch.Tensor, # shape (M, D):  centers of B_j
#     rBs: torch.Tensor        # shape (M, 1) or (M,): radii of B_j
# ) -> torch.Tensor:
#     """
#     Compute the kernelized asymmetric coverage matrix Corr of shape (B, M),
#     where
#       Corr[i, j] = (1/N) * sum_k exp( - max(0, ||X[i,k] - centers_B[j]|| - rB[j]) / rB[j] ).
#
#     Args:
#       X          : Tensor of shape (B, N, D)
#       centers_B  : Tensor of shape (M, D)
#       rBs        : Tensor of shape (M, 1) or (M,)
#
#     Returns:
#       Corr       : Tensor of shape (B, M)
#     """
#     B, N, D = X.shape
#     M = centers_B.shape[0]
#
#     # Ensure radii is shape (M,)
#     rB = rBs.view(-1)  # (M,)
#
#     # Expand for pairwise distances: (B, N, 1, D) - (1, 1, M, D) -> (B, N, M, D)
#     X_exp   = X.unsqueeze(2)                 # (B, N, 1, D)
#     CB_exp  = centers_B.view(1, 1, M, D)     # (1, 1, M, D)
#     dists   = torch.norm(X_exp - CB_exp, dim=3)  # (B, N, M)
#
#     # Compute the "outside" distance beyond the sphere surface
#     rB_exp  = rB.view(1, 1, M)              # (1, 1, M)
#     # Kernel: exponential decay
#     K = torch.exp(-0.05 * (dists - rB_exp) / rB_exp)       # (B, N, M)
#     # Average over the N samples
#     Corr = K.mean(dim=1)                    # (B, M)
#
#     return Corr

def  Sphere_dist1(
    X: torch.Tensor,         # shape (B, N, D): A_i 内的采样点
    centers_B: torch.Tensor, # shape (B, D):  B_j 的中心
    rBs: torch.Tensor        # shape (B, 1) or (B,): B_j 的半径
) -> torch.Tensor:
    """
    返回 Corr 矩阵，形状 (B, B)，其中
      Corr[i,j] = (1/N) Σ_k exp(-max(0, ||X[i,k]-C_B[j]|| - rB_j) / rB_j).
    保证对角线全 1，非重叠时也 > 0。
    """
    B, N, D = X.shape

    # 调整形状以便广播
    centers_B = centers_B.view(1, 1, B, D)   # (1,1,B,D)
    rB = rBs.view(1, 1, B)                   # (1,1,B)

    # 计算样本点到所有 B_j 中心的距离 (B,N,B)
    dists = torch.norm(
        X.unsqueeze(2) - centers_B,
        dim=3
    )

    # 计算“超出部分”并做指数衰减
    # outside = torch.clamp(dists - rB, min=0.0)  # (B,N,B)
    outside = dists - rB # (B,N,B)
    K = torch.exp(-outside * 0.05 / rB)               # (B,N,B)

    # 对 N 个样本点求平均，得到最终矩阵
    Corr, _ = K.max(axis=1)                        # (B,B)
    return Corr

def Sphere_dist(sample1_mean, sample1_radio, sample2_mean, sample2_radio, eps=1e-6):
    # mean_cdist = torch.einsum('ij,kj->ik', sample1_mean, sample2_mean)
    radio_cdist = torch.cdist(sample1_radio, sample2_radio, 2)
    mean_cdist = torch.cdist(sample1_mean, sample2_mean, 2)
    Sdist =  torch.exp(- mean_cdist * radio_cdist)
    return Sdist

def soft_contrastive_nll(logit, matched):
    r"""Compute the negative log-likelihood of the soft contrastive loss.

    .. math::
        NLL_{ij} = -\log p(m = m_{ij} | z_i, z_j)
                 = -\log \left[ \mathbb{I}_{m_{ij} = 1} \sigma(-a \| z_i - z_j \|_2 + b)
                         +  \mathbb{I}_{m_{ij} = -1} (1 - \sigma(-a \| z_i - z_j \|_2 + b)) \right].

    Note that the matching indicator {m_ij} is 1 if i and j are matched otherwise -1.
    Here we define the sigmoid function as the following:
    .. math::
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    Here we sample "logit", s_{ij} by Monte-Carlo sampling to get the expected soft contrastive loss.
    .. math::
        s_{ij}^k = -a \| z_i^k - z_j^k \|_2 + b, z_i^k ~ \mathcal N (\mu_i, \Sigma_i), z_j^k ~ \mathcal N (\mu_j, \Sigma_j).

    Then we can compute NLL by logsumexp (here, we omit `k` in s_{ij}^k for the simplicity):
    .. math::
        NLL_{ij} = -\log \left[ \frac{1}{K^2} \sum_{s_{ij}} \left{ \frac{\exp(s_{ij} m_ij)}{\exp(s_{ij}) + \exp(-s_{ij})} \right} \right]
                 = (\log K^2) -\log \sum_{s_{ij}} \left[ \exp \left( s_{ij} m_ij - \log(\exp(s_{ij} + (-s_{ij}))) \right) \right]
                 = (\log K^2) -logsumexp( s_{ij} m_{ij} - logsumexp(s_{ij}, -s_{ij}) ).

    Parameters
    ----------
    logit: torch.Tensor (shape: N x K ** 2)
    matched: torch.Tensor (shape: N), an element should be either 1 (matched) or -1 (mismatched)

    Returns
    -------
    NLL loss: torch.Tensor (shape: N), should apply `reduction` operator for the backward operation.
    """
    if len(matched.size()) == 1:
        matched = matched[:, None]
    return -(
        (logit * matched - torch.stack(
            (logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)
         ).logsumexp(dim=1)) + np.log(logit.size(1))


class MCSoftContrastiveLoss(nn.Module):
    r"""Creates a criterion that measures the pairwise soft contrastive loss given
    input tensor pairs :math:`X`, :math:`Y` where each tensor is already sampled from a distribution.

    .. math::
        \log p(m = \hat m | x, y)
        p(m = 1 | x, y) = \sigma(-a \| x - y \|_2 + b)
        p(m = 0 | x, y) = 1 - \sigma(-a \| x - y \|_2 + b)
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    This code assumes that :math:`x_i` and :math:`y_j` are in same class if i = j,
    and in different class otherwise.

    The division by :math:`n` can be avoided if sets ``reduction = 'sum'``.

    Parameters
    ----------
    TBD

    Shape
    -----
    Input1 : torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Input2: torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Output: torch.Tensor
        If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """
    def __init__(self, config, reduction='sum'):
        super().__init__()
        if reduction not in {'mean', 'sum', None}:
            raise ValueError('unknown reduction {}'.format(reduction))
        self.reduction = reduction
        self.cLoss = ContrastiveLoss(config, margin=0.2)

        shift = config.init_shift * torch.ones(1)
        negative_scale = config.init_negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.num_samples = config.num_samples

        self.uniform_lambda = config.get('uniform_lambda', 0)
        self.vib_beta = config.get('vib_beta', 0)

    def uniform_loss(self, x, max_samples=16384, t=2):
        if len(x) ** 2 > max_samples:
            # prevent CUDA error: https://github.com/pytorch/pytorch/issues/22313
            indices = np.random.choice(len(x), int(np.sqrt(max_samples)))
            x = x[indices]
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()

    def pairwise_sampling(self, anchors, candidates):
        N = len(anchors)
        if len(anchors) != len(candidates):
            raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
        anchor_idx, selected_idx, matched = self.full_sampling(N)

        anchor_idx = torch.from_numpy(np.array(anchor_idx)).long()
        selected_idx = torch.from_numpy(np.array(selected_idx)).long()
        matched = torch.from_numpy(np.array(matched)).float()

        anchor_idx = anchor_idx.to(anchors.device)
        selected_idx = selected_idx.to(anchors.device)
        matched = matched.to(anchors.device)

        anchors = anchors[anchor_idx]
        selected = candidates[selected_idx]

        cdist = batchwise_cdist(anchors, selected)

        return cdist, matched

    def full_sampling(self, N):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i)
                selected.append(j)
                if i == j:
                    matched.append(1)
                else:
                    matched.append(-1)
        return candidates, selected, matched

    def _compute_loss(self, input1, input2):
        """
        Shape
        -----
        Input1 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Input2: torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Output: torch.Tensor
            If :attr:`reduction` is ``'none'``, then :math:`(N)`.
        """
        distance, matched = self.pairwise_sampling(input1, input2)
        logits = -self.negative_scale * distance + self.shift

        idx = matched == 1
        loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        idx = matched != 1
        loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()

        return {
            'loss': loss_pos + loss_neg,
            'pos_loss': loss_pos,
            'neg_loss': loss_neg,
        }
    # 用于计算最后的图像文本相关性
    def match_prob(self, image_features, caption_features):
        sampled_image_features, sampled_caption_features = image_features, caption_features
        distance = batchwise_cdist(sampled_image_features, sampled_caption_features)

        distance = distance.to(self.negative_scale.device)
        distance = distance.float()
        logits = -self.negative_scale * distance + self.shift
        prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))
        result, _ = prob.max(axis=1)

        # return prob.mean(axis=1)
        return result

    def forward(self, image_features, caption_features, caption_mean, caption_radio, image_mean, image_radio, **kwargs):

        sampled_image_features, sampled_caption_features = image_features, caption_features
        #
        # sphere_cdist = Sphere_dist(caption_mean, caption_radio, image_mean, image_radio)

        # parameter \gamma is set to 0.1 for COCO, and 0.5 for CUB. The results in paper is fused (t2i+i2t)

        # i2t direction
        # sphere_cdist = Sphere_dist1(sampled_caption_features, image_mean, image_radio)
        # cLoss = self.cLoss(sphere_cdist)
        # i2t_loss = self._compute_loss(sampled_image_features, sampled_caption_features)
        # loss = i2t_loss['loss'] + 0.1 * cLoss
        # loss_dict = {'loss': loss.item(),
        #              'PLoss': i2t_loss['loss'].item(),
        #              'cLoss': cLoss.item()}
        # loss = i2t_loss['loss']
        # loss_dict = {'loss': loss.item(),
        #              'PLoss': i2t_loss['loss'].item()
        #            }

        # # t2i direction
        sphere_cdist = Sphere_dist1(sampled_image_features, caption_mean, caption_radio)
        cLoss = self.cLoss(sphere_cdist)
        t2i_loss = self._compute_loss(sampled_caption_features, sampled_image_features)
        loss = t2i_loss['loss'] + 0.1 * cLoss
        loss_dict = {'loss': loss.item(),
                     'PLoss':t2i_loss['loss'].item(),
                     'cLoss': cLoss.item()}

        # loss = t2i_loss['loss']
        # loss_dict = {'loss': loss.item(),
        #              'PLoss': t2i_loss['loss'].item()
        #              }

        return loss, loss_dict
