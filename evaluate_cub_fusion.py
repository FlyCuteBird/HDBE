"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import fire
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import numpy as np
from logger import PythonLogger
from config import parse_config
from datasets import prepare_cub_dataloaders
from engine import TrainerEngine
from engine import CUBEvaluator
from utils.tensor_utils import to_numpy
from criterions.probemb import Sphere_dist
from functools import partial
from tqdm import tqdm

def batch(iterable, batch_size=1):
    """a batch generator
    """
    n_items = len(iterable)
    for batch_idx in range(0, n_items, batch_size):
        yield iterable[batch_idx:min(batch_idx + batch_size, n_items)]


def recall_at_k(knn_labels, gt_labels, k):
    curr_knn_labels = knn_labels[:, :k]
    accuracy_per_sample = np.any(curr_knn_labels == gt_labels, axis=1)

    return np.mean(accuracy_per_sample)


def precision_at_k(knn_labels, gt_labels, k):
    curr_knn_labels = knn_labels[:, :k]
    accuracy_per_sample = np.sum(curr_knn_labels == gt_labels, axis=1) / k
    return np.mean(accuracy_per_sample)


def get_relevance_mask(shape, gt_labels, label_counts, embeddings_come_from_same_source):
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k, max_column in label_counts.items():
        matching_rows = np.where(gt_labels == k)[0]
        if embeddings_come_from_same_source:
            max_column = max_column - 1
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask


def r_precision(knn_labels, gt_labels, embeddings_come_from_same_source, label_counts):
    relevance_mask = get_relevance_mask(
        knn_labels.shape, gt_labels, label_counts, embeddings_come_from_same_source
    )
    matches_per_row = np.sum(
        (knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1
    )
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = matches_per_row / max_possible_matches_per_row
    return accuracy_per_sample

@torch.no_grad()
def batchwise_cdist(samples1, samples2, eps=1e-6):

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
@torch.no_grad()
def match_prob(image_features, caption_features):
    sampled_image_features, sampled_caption_features = image_features, caption_features
    distance = batchwise_cdist(sampled_image_features, sampled_caption_features)

    # distance = distance.to(self.negative_scale.device)
    distance = distance.float()
    logits = -5.0 * distance + 5.0
    prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))
    result, _ = prob.max(axis=1)

    return result
@torch.no_grad()
class ParallelMatMulModule(nn.Module):
    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = q_features.mm(self.g_features)

        if n_embeddings > 1:
            sims = sims.view(int(len(q_features) / n_embeddings),
                             n_embeddings,
                             int(self.g_features.size()[-1] / n_embeddings),
                             n_embeddings)
            sims = sims.permute(0, 1, 3, 2)

            if reduction == 'sum':
                sims = torch.sum(torch.sum(sims, axis=1), axis=1)
            elif reduction == 'max':
                sims = torch.max(torch.max(sims, axis=1)[0], axis=1)[0]

        if embeddings_come_from_same_source:
            # ignore self
            for idx, qidx in enumerate(q_indices):
                sims[idx, qidx] = -np.inf
        return sims
        # sims, pred_ranks = (-sims).sort()
        # return sims, pred_ranks
@torch.no_grad()
class MatchingSphereModule(nn.Module):
    def forward(self, sample1_mean, sample1_radio, sample2_mean, sample2_radio):
        return Sphere_dist(sample1_mean, sample1_radio, sample2_mean, sample2_radio)


@torch.no_grad()
class MatchingProbModule(nn.Module):

    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = torch.zeros(len(q_features), len(self.g_features))
        # print(sims.shape)
        for idx, q_feature in enumerate(q_features):
            _sim = match_prob(q_feature.unsqueeze(0), self.g_features)
            sims[idx] = _sim
        return sims


def elk_dist(mu1, mu2, log_sigma1, log_sigma2):
    sum_sigma_sq = torch.exp(2 * log_sigma1) + torch.exp(2 * log_sigma2)
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    elk = dist / (sum_sigma_sq) + torch.log(sum_sigma_sq)
    return -0.5 * torch.sum(elk, dim=1)


def kl_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)

    var2 = torch.exp(log_sigma2 * 2)
    dist = dist / var2 + 2 * (log_sigma1 - log_sigma2) + torch.exp(log_sigma1 * 2) / var2
    return -torch.sum(dist, dim=1)


def reverse_kl_dist(mu1, mu2, log_sigma1, log_sigma2):
    return kl_dist(mu2, mu1, log_sigma2, log_sigma1)


def js_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)

    var1 = torch.exp(log_sigma1 * 2)
    var2 = torch.exp(log_sigma2 * 2)
    dist = (dist + var1) / var2 + (dist + var2) / var1
    return -torch.sum(dist, dim=1)


def bhattacharyya_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    sigma1 = torch.exp(log_sigma1)
    sigma2 = torch.exp(log_sigma2)

    dist = dist / (torch.exp(log_sigma1 * 2) + torch.exp(log_sigma2 * 2))
    dist = dist + 2 * torch.log(sigma1 / sigma2 + sigma2 / sigma1)
    ddd = 2 * torch.log(torch.ones(1) * 2)
    dist = dist.float() - ddd.to(dist.device).float()
    dist = dist / 4
    return -torch.sum(dist, dim=1)


def wasserstein_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    dist = dist + (torch.exp(log_sigma1) - torch.exp(log_sigma2)) ** 2
    return -torch.sum(dist, dim=1)


def l2_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    return -torch.sum(dist, dim=1)


def min_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    return -torch.min(dist, dim=1)[0]


def max_dist(mu1, mu2, log_sigma1, log_sigma2):
    dist = (mu1 - mu2) ** 2
    dist = dist.squeeze(1)
    return -torch.max(dist, dim=1)[0]

@torch.no_grad()
class HybridDistanceModule(nn.Module):
    def __init__(self, dist_fn_name):
        super().__init__()
        print(dist_fn_name)
        if dist_fn_name == 'elk':
            self.dist_fn = elk_dist
        elif dist_fn_name == 'l2':
            self.dist_fn = l2_dist
        elif dist_fn_name == 'min':
            self.dist_fn = min_dist
        elif dist_fn_name == 'max':
            self.dist_fn = max_dist
        elif dist_fn_name == 'wasserstein':
            self.dist_fn = wasserstein_dist
        elif dist_fn_name == 'kl':
            self.dist_fn = kl_dist
        elif dist_fn_name == 'reverse_kl':
            self.dist_fn = reverse_kl_dist
        elif dist_fn_name == 'js':
            self.dist_fn = js_dist
        elif dist_fn_name == 'bhattacharyya':
            self.dist_fn = bhattacharyya_dist
        else:
            raise ValueError(dist_fn_name)

    def set_g_features(self, g_features, g_sigmas=None):
        self._g_features = g_features
        self.g_features = None
        self._g_sigmas = torch.from_numpy(g_sigmas)
        self.g_sigmas = None

    def forward(self, q_features, n_embeddings=1, reduction=None,
                embeddings_come_from_same_source=False,
                q_indices=None, q_sigmas=None):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
            self.g_sigmas = self._g_sigmas.to(q_features.device)

        sims = torch.zeros(len(q_features), len(self.g_features))

        q_sigmas = torch.from_numpy(q_sigmas).to(q_features.device)
        for idx, (q_feature, q_sigma) in enumerate(zip(q_features, q_sigmas)):
            _sim = self.dist_fn(q_feature.unsqueeze(0), self.g_features, q_sigma.unsqueeze(0), self.g_sigmas)
            sims[idx] = _sim
        return sims

@torch.no_grad()
def get_scores(knn_labels, gt_labels, best_pred_ranks,
               label_counts, embeddings_come_from_same_source,
               sigmas=None, dismatch_probs=None):
    # https://github.com/KevinMusgrave/pytorch-metric-learning/blob/0b575b556fe339c2a62043d0ff0efe7fe85107bc/src/pytorch_metric_learning/utils/accuracy_calculator.py#L45
    relevance_mask = get_relevance_mask(
        knn_labels.shape, gt_labels, label_counts, embeddings_come_from_same_source
    )
    num_samples, num_k = knn_labels.shape
    equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k + 1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx
    summed_precision_per_row = np.sum(precision_at_ks * relevance_mask, axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    mean_average_precision_at_r = np.mean(summed_precision_per_row / max_possible_matches_per_row)

    pr_per_sample = r_precision(knn_labels, gt_labels, embeddings_come_from_same_source, label_counts)
    precision_r = np.mean(pr_per_sample)

    sigma_vs_map = None
    dismatchprob_vs_map = None

    recall_1 = recall_at_k(knn_labels, gt_labels, 1)
    recall_5 = recall_at_k(knn_labels, gt_labels, 5)
    recall_10 = recall_at_k(knn_labels, gt_labels, 10)

    precision_5 = precision_at_k(knn_labels, gt_labels, 5)
    precision_10 = precision_at_k(knn_labels, gt_labels, 10)
    precision_50 = precision_at_k(knn_labels, gt_labels, 50)
    precision_100 = precision_at_k(knn_labels, gt_labels, 100)

    scores = {
        'recall_1': recall_1,
        'recall_5': recall_5,
        'recall_10': recall_10,
        'rsum': recall_1 + recall_5 + recall_10,
        'precision_5': precision_5,
        'precision_10': precision_10,
        'precision_50': precision_50,
        'precision_100': precision_100,
        'r_precision': precision_r,
        'map_r': mean_average_precision_at_r,
    }

    if sigma_vs_map:
        scores['sigma_vs_map'] = sigma_vs_map
        scores['dismatchprob_vs_map'] = dismatchprob_vs_map

    if best_pred_ranks is not None:
        medr = np.floor(np.median(best_pred_ranks)) + 1
        meanr = np.mean(best_pred_ranks) + 1
        scores['medr'] = medr
        scores['meanr'] = meanr
    return scores

@torch.no_grad()
def evaluate_recall(q_features, g_features, q_mean, g_mean, q_radio, g_radio,
                    q1_features, g1_features, q1_mean, g1_mean, q1_radio, g1_radio,
                    q_labels, g_labels,
                    q_ids=None, g_ids=None,
                    q_classes=None, g_classes=None,
                    label_counts=None,
                    embeddings_come_from_same_source=False,
                    q_sigmas=None, g_sigmas=None,
                    reduction='sum', batch_size=256
                    ):
    """Evaluate recall

    Args:
        q_features (tensor): N_q x d query features
        g_features (tensor): N_g x d gallery features
        q_labels (tensor): N query labels
        g_labels (tensor): N gallery labels
    """
    eval_method = 'matching_prob'

    if len(q_features) != len(q_labels):
        raise RuntimeError('length mismatch {}, {}'.format(q_features.shape,
                                                           q_labels.shape))
    if len(g_features) != len(g_labels):
        raise RuntimeError('length mismatch {}, {}'.format(g_features.shape,
                                                           g_labels.shape))
    pbar = partial(tqdm, disable=not False)
    n_queries = len(q_labels)
    n_galleries = len(g_labels)
    best_pred_ranks = np.zeros(n_queries)
    MSM = MatchingSphereModule()

    if eval_method == 'matmul':
        print('eval method: matmul:')
        pmm = ParallelMatMulModule()
        pmm1 = ParallelMatMulModule()
        g_features = g_features.view(n_galleries * 7, -1).t()
        g1_features = g1_features.view(n_galleries * 7, -1).t()
    elif eval_method == 'matching_prob':
        print('eval method: matching_prob:')
        pmm = MatchingProbModule()
        pmm1 = MatchingProbModule()
    else:
        print('eval method: others:')
        pmm = HybridDistanceModule(eval_method)
        pmm1 = HybridDistanceModule(eval_method)
    pmm.set_g_features(g_features, g_sigmas)
    pmm1.set_g_features(g1_features, g_sigmas)

    q_features = q_features.to("cuda")
    q_mean = q_mean.to("cuda")
    q_radio = q_radio.to("cuda")
    g_mean = g_mean.to("cuda")
    g_radio = g_radio.to("cuda")

    q1_features = q1_features.to("cuda")
    q1_mean = q1_mean.to("cuda")
    q1_radio = q1_radio.to("cuda")
    g1_mean = g1_mean.to("cuda")
    g1_radio = g1_radio.to("cuda")

    sigmas = None
    dismatch_probs = None
    if q_sigmas is not None and len(q_sigmas.shape) == 2:
        sigmas = []
        dismatch_probs = []

    knn_labels = []
    gt_labels = []
    for q_indices in pbar(batch(range(n_queries), batch_size=batch_size)):
        q_indices = np.array(q_indices)

        if eval_method != 'matching_prob':
            _q_feature = q_features[q_indices, :]
            _q_feature = _q_feature.view(len(q_indices) * 7, -1)
            _q_mean = q_mean[q_indices, :]
            _q_radio = q_radio[q_indices, :]

            _q1_feature = q1_features[q_indices, :]
            _q1_feature = _q1_feature.view(len(q_indices) * 7, -1)
            _q1_mean = q1_mean[q_indices, :]
            _q1_radio = q1_radio[q_indices, :]
        else:
            _q_feature = q_features[q_indices, :, :]
            _q_mean = q_mean[q_indices, :]
            _q_radio = q_radio[q_indices, :]

            _q1_feature = q1_features[q_indices, :, :]
            _q1_mean = q1_mean[q_indices, :]
            _q1_radio = q1_radio[q_indices, :]

        if q_sigmas is not None:
            _q_sigmas = q_sigmas[q_indices]
        else:
            _q_sigmas = None
        # sims = pmm(_q_feature, n_embeddings=7,
        #            reduction=reduction, embeddings_come_from_same_source=embeddings_come_from_same_source,
        #            q_indices=q_indices, q_sigmas=_q_sigmas)
        if eval_method == 'matching_prob':
            sims = pmm(_q_feature, n_embeddings=7)
            sims1 = pmm1(_q1_feature, n_embeddings=7)
        else:
            sims = pmm(_q_feature, n_embeddings=7,
                       reduction=reduction, embeddings_come_from_same_source=embeddings_come_from_same_source,
                       q_indices=q_indices, q_sigmas=_q_sigmas)
            sims1 = pmm1(_q1_feature, n_embeddings=7,
                       reduction=reduction, embeddings_come_from_same_source=embeddings_come_from_same_source,
                       q_indices=q_indices, q_sigmas=_q_sigmas)
        # MSM_result = MSM(_q_mean, _q_radio, g_mean, g_radio).to(sims.device)
        # MSM_result1 = MSM(_q1_mean, _q1_radio, g1_mean, g1_radio).to(sims.device)
        fusion_sims = (sims + sims1)
        _, pred_ranks = (-fusion_sims).sort()

        _, sorted_indices = pred_ranks.sort()
        for idx, sorted_db_index in enumerate(sorted_indices):
            _, _sorted_idx = sorted_db_index.sort()
            q_class = int(q_classes[q_indices[idx]].item())
            r_classes = [int(item) for item in
                         g_classes[to_numpy(_sorted_idx)[0]].tolist()]
            gt_labels.append(q_class)
            knn_labels.append(r_classes)

        for idx, q_idx in enumerate(q_indices):
            pos_indices = np.where(g_labels == q_labels[q_idx])[0]
            _pred_ranks = [torch.where(pred_ranks[idx] == pos_idx)[0][0].item() for pos_idx in pos_indices]
            best_pred_ranks[q_idx] = min(_pred_ranks)

    knn_labels = np.array(knn_labels)
    gt_labels = np.array(gt_labels)
    gt_labels = gt_labels[:, None]

    scores = get_scores(knn_labels, gt_labels, best_pred_ranks,
                             label_counts, embeddings_come_from_same_source,
                             sigmas, dismatch_probs)

    return scores

@torch.no_grad()
def result(engine1_features, engine2_features, dataloader, n_crossfolds=None,
             reduction='sum', eval_batch_size=256,
             report_sigma=False,
             key=None):
    """evaluate image-to-caption and caption-to-image retrieval tasks.
    """
    scores = {}

    # if self.logger:
    #     self.logger.log('extracting features...')
    #
    # extracted_features = self.extract_features(dataloader)
    #
    # image_features = extracted_features['image_features']
    # caption_features = extracted_features['caption_features']
    # image_mean = extracted_features['image_mean']
    # caption_mean = extracted_features['caption_mean']
    # image_radio = extracted_features['image_radio']
    # caption_radio = extracted_features['caption_radio']
    # image_sigmas = extracted_features['image_sigmas']
    # caption_sigmas = extracted_features['caption_sigmas']
    # image_classes = extracted_features['image_classes']
    # caption_classes = extracted_features['caption_classes']

    ###
    image_features1 = engine1_features['image_features']
    print('image_features1', image_features1.shape)
    caption_features1 = engine1_features['caption_features']
    print('caption_features1', caption_features1.shape)
    image_mean1 = engine1_features['image_mean']
    caption_mean1 = engine1_features['caption_mean']
    image_radio1 = engine1_features['image_radio']
    caption_radio1 = engine1_features['caption_radio']
    image_sigmas1 = engine1_features['image_sigmas']
    caption_sigmas1 = engine1_features['caption_sigmas']
    image_classes = engine1_features['image_classes']
    caption_classes = engine1_features['caption_classes']

    image_features2 = engine2_features['image_features']
    caption_features2 = engine2_features['caption_features']
    image_mean2 = engine2_features['image_mean']
    caption_mean2 = engine2_features['caption_mean']
    image_radio2 = engine2_features['image_radio']
    caption_radio2 = engine2_features['caption_radio']
    image_sigmas2 = engine2_features['image_sigmas']
    caption_sigmas2 = engine2_features['caption_sigmas']
    image_classes2 = engine2_features['image_classes']
    caption_classes2 = engine2_features['caption_classes']
    ####


    scores['image_sigma'] = np.mean(image_sigmas1)
    scores['caption_sigma'] = np.mean(caption_sigmas1)

    q_sigmas = None
    g_sigmas = None

    if report_sigma:
        q_sigmas = image_sigmas1
        g_sigmas = caption_sigmas1

    print('evaluating i2t...')
    label_counts = {k: len(v) for k, v in dataloader.dataset.class_to_indices.items()}
    scores['i2t'] = evaluate_recall(image_features1,
                                    caption_features1,
                                    image_mean1,
                                    caption_mean1,
                                    image_radio1,
                                    caption_radio1,
                                    image_features2,
                                    caption_features2,
                                    image_mean2,
                                    caption_mean2,
                                    image_radio2,
                                    caption_radio2,
                                    image_classes,
                                    caption_classes,
                                    image_classes,
                                    caption_classes,
                                    image_classes,
                                    caption_classes,
                                    label_counts,
                                    reduction=reduction,
                                    q_sigmas=q_sigmas,
                                    g_sigmas=g_sigmas,
                                    batch_size=eval_batch_size)

    if report_sigma:
        q_sigmas = caption_sigmas1
        g_sigmas = image_sigmas1

    print('evaluating t2i...')
    label_counts = {k: len(v) for k, v in dataloader.dataset.class_to_img_indices.items()}
    scores['t2i'] = evaluate_recall(caption_features1,
                                    image_features1,
                                    caption_mean1,
                                    image_mean1,
                                    caption_radio1,
                                    image_radio1,
                                    caption_features2,
                                    image_features2,
                                    caption_mean2,
                                    image_mean2,
                                    caption_radio2,
                                    image_radio2,
                                    caption_classes,
                                    image_classes,
                                    caption_classes,
                                    image_classes,
                                    caption_classes,
                                    image_classes,
                                    label_counts,
                                    reduction=reduction,
                                    q_sigmas=q_sigmas,
                                    g_sigmas=g_sigmas,
                                    batch_size=eval_batch_size)
    for key in ('rsum', 'medr', 'meanr'):
        scores[key] = scores['i2t'][key] + scores['t2i'][key]
    return scores


@torch.no_grad()
def evaluate(config, dataset_name, model_path1, model_path2, dataloader, vocab, logger):
    logger.log('start evaluation')
    engine1 = TrainerEngine()
    engine2 = TrainerEngine()

    engine1.set_logger(logger)
    engine2.set_logger(logger)

    evaluator1 = CUBEvaluator(eval_method=config.model.get('eval_method', 'matmul'),
                             verbose=True,
                             eval_device='cuda')
    evaluator2 = CUBEvaluator(eval_method=config.model.get('eval_method', 'matmul'),
                              verbose=True,
                              eval_device='cuda')

    engine1.create(config, vocab.word2idx, evaluator1)
    engine2.create(config, vocab.word2idx, evaluator2)

    engine1.load_models(model_path1,
                       load_keys=['model', 'criterion'])
    engine1.load_models(model_path2,
                        load_keys=['model', 'criterion'])
    logger.log('extracting features...')
    engine1_features = engine1.evaluator.extract_features(dataloader)
    engine2_features = engine2.evaluator.extract_features(dataloader)
    if not isinstance(dataloader, dict):
        val_loaders = {'te': dataloader}
    Scores = {}
    for key, data_loader in val_loaders.items():
        logger.log('Evaluating {}...'.format(key))
        Scores[key] = result(engine1_features, engine2_features, data_loader, n_crossfolds=-1, key=key)

    logger.pretty_log_dict(Scores)
    return Scores

@torch.no_grad()
def main(config_path='./config/cub/pcme_cub.yaml',
         dataset_root='datasets',
         caption_root='datasets/annotations/cub/cub_captions',
         model_path1='t2i_0.05_0_model_last.pth',
         model_path2='i2t_0.05_0_model_last.pth',
         dataset_name='cub',
         split='val',
         vocab_path='datasets/vocabs/cub_vocab.pkl',
         cache_dir='./.vector_cache/',
         dump_to=None,
         **kwargs):
    config = parse_config(config_path,
                          strict_cast=False,
                          model__cache_dir=cache_dir,
                          **kwargs)

    logger = PythonLogger()
    logger.log('preparing data loaders..')
    dataloaders, vocab = prepare_cub_dataloaders(config.dataloader,
                                                 dataset_name,
                                                 dataset_root,
                                                 caption_root,
                                                 vocab_path)
    scores = evaluate(config, dataset_name, model_path1, model_path2, dataloaders[split], vocab, logger)
    if dump_to:
        torch.save(scores, dump_to)


if __name__ == '__main__':
    fire.Fire(main)
