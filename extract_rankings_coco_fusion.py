"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from logger import PythonLogger
import datetime
import fire
from config import parse_config
from datasets import prepare_coco_dataloaders
from engine import COCOEvaluator
from engine import COCORetrievalEngine
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.tensor_utils import to_numpy
from criterions.probemb import Sphere_dist, Sphere_dist1

def batch(iterable, batch_size=1):
    """a batch generator
    """
    n_items = len(iterable)
    for batch_idx in range(0, n_items, batch_size):
        yield iterable[batch_idx:min(batch_idx + batch_size, n_items)]


def recall_at_k(ranks, k):

    return 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
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
    logits = -15.0 * distance + 15.0
    prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))

    result, _ = prob.max(axis=1)

    return result
@torch.no_grad()
class ParallelMatMulModule(nn.Module):
    def set_g_features(self, g_features):
        self._g_features = g_features
        self.g_features = None

    def forward(self, q_features, n_embeddings=1):
        if self.g_features is None:
            self.g_features = self._g_features.to(q_features.device)
        sims = q_features.mm(self.g_features)

        if n_embeddings > 1:
            sims = sims.view(int(len(q_features) / n_embeddings),
                             n_embeddings,
                             int(self.g_features.size()[-1] / n_embeddings),
                             n_embeddings)
            sims = sims.permute(0, 1, 3, 2)
            sims = torch.sum(torch.sum(sims, axis=1), axis=1)
        return sims

        # sims, pred_ranks = (-sims).sort()
        # return sims, pred_ranks
@torch.no_grad()
class MatchingSphereModule(nn.Module):
    # def forward(self, sample1_mean, sample1_radio, sample2_mean, sample2_radio):
    #     return Sphere_dist1(sample1_mean, sample1_radio, sample2_mean, sample2_radio)

    def forward(self, sample1_features, sample2_mean, sample2_radio):
        return Sphere_dist1(sample1_features, sample2_mean, sample2_radio)


@torch.no_grad()
class MatchingProbModule(nn.Module):

    def set_g_features(self, g_features):
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
        # sims, pred_ranks = (-sims).sort()
        # return sims, pred_ranks

@torch.no_grad()
def retrieve(q_features, q_mean, q_radio,
             g_features, g_mean, g_radio,
             q1_features, q1_mean, q1_radio,
             g1_features, g1_mean, g1_radio,
             q_ids, g_ids,
             q_modality,
             q_classes=None, g_classes=None,
             topk=10,
             batch_size=1024):
    MSM = MatchingSphereModule()
    pbar = partial(tqdm, disable=not True)
    if len(q_features) != len(q_ids):
        raise RuntimeError('length mismatch {}, {}'.format(q_features.shape,
                                                           q_ids.shape))
    if len(g_features) != len(g_ids):
        raise RuntimeError('length mismatch {}, {}'.format(g_features.shape,
                                                           g_ids.shape))
    if isinstance(q_ids, list) or isinstance(g_ids, list):
        q_ids = np.array(q_ids)
        g_ids = np.array(g_ids)
    n_queries = len(q_ids)
    n_galleries = len(g_ids)

    # pmm = ParallelMatMulModule()
    # g_features = g_features.view(n_galleries * 7, -1).t()

    pmm = MatchingProbModule()
    pmm.set_g_features(g_features)
    pmm1 = MatchingProbModule()
    pmm1.set_g_features(g1_features)

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

    retrieved_items = {}
    retrieved_scores = {}

    for q_indices in pbar(batch(range(n_queries), batch_size=batch_size)):
        q_indices = np.array(q_indices)

        _q_feature = q_features[q_indices, :]
        _q_mean = q_mean[q_indices, :]
        _q_radio = q_radio[q_indices, :]
        # _q_feature = _q_feature.view(len(q_indices) * 7, -1)

        _q1_feature = q1_features[q_indices, :]
        _q1_mean = q1_mean[q_indices, :]
        _q1_radio = q1_radio[q_indices, :]
        # _q1_feature = _q1_feature.view(len(q_indices) * 7, -1)
        sims = pmm(_q_feature, n_embeddings=7)
        sim1 = pmm1(_q1_feature, n_embeddings=7)
        # MSM_result = MSM(_q_mean, _q_radio, g_mean, g_radio).to(sims.device)
        # MSM_result1 = MSM(_q1_mean, _q1_radio, g1_mean, g1_radio).to(sims.device)
        # MSM_result = MSM(_q_feature, g_mean, g_radio).to(sims.device)
        # MSM_result1 = MSM(_q1_feature, g1_mean, g1_radio).to(sims.device)
        # fusion_sims0 = (sims + 0.1 * MSM_result)
        # fusion_sims1 = (sim1 + 0.1 * MSM_result1)
        # fusion_sims = fusion_sims0 + fusion_sims1
        fusion_sims = sims + sim1
        _, pred_ranks = (-fusion_sims).sort()

        _, sorted_indices = pred_ranks.sort()

        for idx, sorted_db_index in enumerate(sorted_indices):
            _, _sorted_idx = sorted_db_index.sort()
            retrieved_items[q_ids[q_indices[idx]]] = [item for item in g_ids[to_numpy(_sorted_idx)[0][:]]]
            retrieved_scores[q_ids[q_indices[idx]]] = fusion_sims[idx][:].cpu().numpy()
    return retrieved_items, retrieved_scores, None

@torch.no_grad()
def retrieve_from_features(q_features, q_mean, q_radio, g_features, g_mean, g_radio, q1_features, q1_mean, q1_radio, g1_features, g1_mean, g1_radio,
                           q_modality, q_classes, g_classes, g_meta_loader=None, q_meta_loader=None, q_gt_loader=None, q_ids=None, g_ids=None, q_sigmas=None, g_sigmas=None, topk=10, batch_size=1024):
    if q_modality not in ['image', 'caption']:
        raise ValueError

    if q_ids is None:
        q_ids = [idx for idx in range(len(q_features))]

    if q_sigmas is None:
        q_sigmas = [0 for _ in range(len(q_features))]

    retrieved_items, scores, metas = retrieve(q_features,
                                              q_mean,
                                              q_radio,
                                              g_features,
                                              g_mean,
                                              g_radio,
                                              q1_features,
                                              q1_mean,
                                              q1_radio,
                                              g1_features,
                                              g1_mean,
                                              g1_radio,
                                              q_ids,
                                              g_ids,
                                              q_classes,
                                              g_classes,
                                              q_modality,
                                              topk=topk,
                                              batch_size=batch_size)

    ret = {}
    for q_sigma, (q_id, items) in zip(q_sigmas, retrieved_items.items()):
        q_id = int(q_id)

        meta = {}
        # print("-----111_items-----:")
        # print(type(items))
        # print(len(items))
        # print(type(items[0]))
        # print(items[0])
        # print('-----111_items_end-----')
        meta['retrieved'] = g_meta_loader(items)
        meta['scores'] = scores[q_id]
        _meta = q_meta_loader(q_id)[0].copy()
        if q_gt_loader:
            _meta.update({'gt_captions': [item['caption'] for item in q_gt_loader[q_id]]})
        else:
            _meta.update({'gt_image': g_meta_loader(_meta['image_id'])[0]})
        meta['query'] = _meta
        # meta['query_sigma'] = q_sigma
        ret[q_id] = meta
    return ret


@torch.no_grad()
def main(config_path,
         dataset_root,
         model_path1,
         model_path2,
         dump_to,
         vocab_path='datasets/vocabs/coco_vocab.pkl',
         cache_dir='.vector_cache',
         dump_features_to=None,
         split='te',
         topk=-1,
         **kwargs):
    dt = datetime.datetime.now()

    config = parse_config(config_path,
                          strict_cast=False,
                          model__cache_dir=cache_dir,
                          **kwargs)

    logger = PythonLogger()
    logger.log('preparing data loaders..')
    dataloaders, vocab = prepare_coco_dataloaders(config.dataloader,
                                                  dataset_root, vocab_path)

    engine = COCORetrievalEngine()
    engine1 = COCORetrievalEngine()
    engine.set_logger(logger)
    engine1.set_logger(logger)

    evaluator = COCOEvaluator(eval_method='matching_prob',
                              verbose=True,
                              eval_device='cuda',
                              n_crossfolds=5)
    evaluator1 = COCOEvaluator(eval_method='matching_prob',
                              verbose=True,
                              eval_device='cuda',
                              n_crossfolds=5)
    engine.create(config, vocab.word2idx, evaluator)
    engine1.create(config, vocab.word2idx, evaluator1)
    engine.load_models(model_path1,
                       load_keys=['model', 'criterion'])
    engine1.load_models(model_path2,
                       load_keys=['model', 'criterion'])
    engine.set_gallery_from_dataloader(dataloaders[split])
    engine1.set_gallery_from_dataloader(dataloaders[split])

    print('-----test_ids-----')
    print('engine.image_ids:', engine.image_ids)
    print('engine1.image_ids:', engine1.image_ids)
    print('engine.caption_ids:', engine.caption_ids)
    print('engine1.caption_ids:', engine1.caption_ids)

    if dump_features_to:
        torch.save({
            'images': engine.image_features,
            'captions': engine.caption_features,
            'image_ids': engine.image_ids,
            'image_sigmas': engine.image_sigmas,
            'image_classes': engine.image_classes,
            'caption_ids': engine.caption_ids,
            'caption_sigmas': engine.caption_sigmas,
            'caption_classes': engine.caption_classes,
        }, dump_features_to)

    i2t_retrieved_items = retrieve_from_features(engine.image_features,
                                                 engine.image_mean,
                                                 engine.image_radio,
                                                 engine.caption_features,
                                                 engine.caption_mean,
                                                 engine.caption_radio,
                                                 engine1.image_features,
                                                 engine1.image_mean,
                                                 engine1.image_radio,
                                                 engine1.caption_features,
                                                 engine1.caption_mean,
                                                 engine1.caption_radio,
                                                 q_modality='image',
                                                 q_classes=engine.image_classes,
                                                 g_classes=engine.caption_classes,
                                                 g_meta_loader=engine.coco.loadAnns,
                                                 q_meta_loader=engine.coco.loadImgs,
                                                 q_gt_loader=engine.coco.imgToAnns,
                                                 q_ids=engine.image_ids,
                                                 g_ids=engine.caption_ids,
                                                 topk=topk,
                                                 batch_size=config.dataloader.eval_batch_size,
                                                )
    t2i_retrieved_items = retrieve_from_features(engine.caption_features,
                                                 engine.caption_mean,
                                                 engine.caption_radio,
                                                 engine.image_features,
                                                 engine.image_mean,
                                                 engine.image_radio,
                                                 engine1.caption_features,
                                                 engine1.caption_mean,
                                                 engine1.caption_radio,
                                                 engine1.image_features,
                                                 engine1.image_mean,
                                                 engine1.image_radio,
                                                 q_modality='caption',
                                                 q_classes=engine.caption_classes,
                                                 g_classes=engine.image_classes,
                                                 g_meta_loader=engine.coco.loadImgs,
                                                 q_meta_loader=engine.coco.loadAnns,
                                                 q_gt_loader=None,
                                                 q_ids=engine.caption_ids,
                                                 g_ids=engine.image_ids,
                                                 topk=topk,
                                                 batch_size=config.dataloader.eval_batch_size,
                                                )

    data = {
        'i2t': i2t_retrieved_items,
        't2i': t2i_retrieved_items,
    }

    torch.save(data, dump_to)
    logger.log('takes {}'.format(datetime.datetime.now() - dt))


if __name__ == '__main__':
    fire.Fire(main)
