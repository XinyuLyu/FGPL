# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. vctree 40.2
import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
import json
from maskrcnn_benchmark.layers.gcn._utils import adj_laplacian
class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = alpha_t
        self.gamma = gamma

    def __call__(self, outputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                    weight=self.alpha_t, reduction='none')
        focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss
def softmax(x,dim):
    return (x-x.logsumexp(dim, keepdim=True)).exp()
def log_softmax(x,dim):
    return x-x.logsumexp(dim,keepdim=True)
def log_weighted_softmax(x,w,dim):
    lower = w.log()+x
    return x- lower.logsumexp(dim, keepdim=True)
def predicate_statistics(predicate_proportion, predicate_count, pred_weight_beta,id2pred):
    if id2pred == None:
        mean_pred = predicate_count[0] * 2
        min_value = 0.01
        predicate_proportion.append((1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** mean_pred))
        for i in range(1,len(predicate_count),1):
            if predicate_count[i]==0:
                predicate_count[i] = min_value
            predicate_proportion.append((1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** predicate_count[i]))
        predicate_proportion = predicate_proportion / np.sum(predicate_proportion) * len(predicate_proportion)
        pred_weight = (torch.FloatTensor(predicate_proportion)).cuda()
        return pred_weight
    else:
        mean_pred = np.sum(np.array(list(predicate_count.values()))) * 2
        predicate_proportion.append((1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** mean_pred))
        for i in range(len(predicate_count)):
            predicate_proportion.append(
                (1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** predicate_count[id2pred[str(i + 1)]]))
        predicate_proportion = predicate_proportion / np.sum(predicate_proportion) * len(predicate_proportion)
        pred_weight = (torch.FloatTensor(predicate_proportion)).cuda()
        return pred_weight#, dict
class MyCrossEntropyLoss(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.

    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, target, factor):
        log_probabilities = log_weighted_softmax(logits,factor,-1)
        return -log_probabilities.index_select(-1, target).diag().mean()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_dis, neg_dis):
        losses = F.relu(neg_dis - pos_dis + self.margin).mean()
        return losses
class TripletLoss_CB(nn.Module):

    def __init__(self, margin):
        super(TripletLoss_CB, self).__init__()
        self.margin = margin

    def forward(self, pos_samp, neg_samp, weights):
        weights_sum = weights.sum()
        losses = ((F.relu(neg_samp - pos_samp + self.margin) * weights).mean(dim=1)).sum() / weights_sum
        return losses
class SoftCrossEntropy(nn.Module):

    def __init__(self, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        self.reduction = reduction

    def softmax_cross_entropy_with_softtarget(self, input, target):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if self.reduction == 'none':
            return batchloss
        elif self.reduction == 'mean':
            return torch.mean(batchloss)
        elif self.reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')
class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
            self,
            attri_on,
            num_attri_cat,
            max_num_attri,
            attribute_sampling,
            attribute_bgfg_ratio,
            use_label_smoothing,
            predicate_proportion,
            use_extra_loss,
            use_logits_reweight,
            use_bce_batch,
            Mitigation_Factor_hyper,
            use_miti,
            miti_tail,
            use_com,
            com_adj,
            use_pcm,
            Compensation_Factor_hyper,
            use_category_reweight,
            use_bce,
            pred_weight_beta,
            use_cde,
            use_focal,
            focal_gamma,
            use_contra_loss,
            use_contra_bce,
            use_contra_distance_loss_value,
            use_contra_distance_loss_cof,
            candidate_number,
            relation_predictor
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5, ] + predicate_proportion)).cuda()
        self.nllloss = nn.NLLLoss()
        ###############################################################              FPGL  Start                   #######################################################################
        # Setting of FGPL
        self.use_extra_loss = use_extra_loss # True-> use FGPL False-> Cross-Entropy Loss
        self.use_contra_distance_loss_value = use_contra_distance_loss_value # margin (0.5 by default)
        self.use_contra_distance_loss_cof = use_contra_distance_loss_cof # hyper-paramter:lambda (0.1 by default)
        self.Mitigation_Factor_hyper = Mitigation_Factor_hyper
        self.Compensation_Factor_hyper = Compensation_Factor_hyper
        self.pred_weight_beta = pred_weight_beta
        self.candidate_number = candidate_number
        self.use_category_reweight = use_category_reweight
        self.use_logits_reweight = use_logits_reweight # True-> use CDL  Re-weight based on Cross-Entropy Loss
        self.use_pcm = use_pcm
        self.use_miti = use_miti
        self.miti_tail = miti_tail
        self.use_com = use_com
        self.com_adj = com_adj
        self.use_bce = use_bce
        self.use_cde = use_cde
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        # Setting of EDL use_contra_loss & use_contra_bce should be True if wanna uses EDL
        self.use_contra_loss = use_contra_loss  # True-> use EDL  False-> do not use EDL
        self.use_contra_bce = use_contra_bce # True-> use EDL  False-> do not use EDL
        self.use_bce_batch = use_bce_batch
        self.relation_predictor = relation_predictor

        predicate_proportion = []
        vg_dict = json.load(open('/home/lvxinyu/lib/scene-graph-benchmark/datasets/vg/VG-SGG-dicts-with-attri.json', 'r'))
        id2pred = vg_dict['idx_to_predicate']
        predicate_count = vg_dict['predicate_count']
        self.pred_weight = predicate_statistics(predicate_proportion, predicate_count, self.pred_weight_beta, id2pred)
        # Load Biased_Predicate_Prediction
        if self.relation_predictor == 'TransformerPredictor':
            # freq-bias has established context-predicate associations, and is used to generate biased predictions of baslines.
            # As we extract baselines' confusion matrix from their biased prediction trained with freq bias, it both consider textual & visual context information for each sample of objects and subjects
            self.pred_adj_np = np.load('/home/lvxinyu/lib/scene-graph-benchmark/misc/conf_mat_transformer_train.npy')
            self.pred_adj_np[0, :] = 0.0
            self.pred_adj_np[:, 0] = 0.0
            self.pred_adj_np[0, 0] = 1.0 # set 1 for ``background'' predicate
            self.pred_adj_np = self.pred_adj_np / (self.pred_adj_np.sum(0)[:, None] + 1e-8)
            self.pred_adj_np = adj_laplacian(self.pred_adj_np) # normalize to [0,1] to get predicate-predicate association
            self.pred_adj_np = torch.from_numpy(self.pred_adj_np).float().cuda()
            self.pred_adj_np_diag = torch.diag(self.pred_adj_np)


        elif self.relation_predictor == 'MotifPredictor':
            self.pred_adj_np = np.load('/home/lvxinyu/lib/scene-graph-benchmark/misc/conf_mat_motif_train.npy')
            
            self.pred_adj_np[0, :] = 0.0
            self.pred_adj_np[:, 0] = 0.0
            self.pred_adj_np[0, 0] = 1.0
            self.pred_adj_np = self.pred_adj_np / (self.pred_adj_np.sum(-1)[:, None] + 1e-8)
            self.pred_adj_np = adj_laplacian(self.pred_adj_np)
            self.pred_adj_np = torch.from_numpy(self.pred_adj_np).float().cuda()
            self.pred_adj_np_diag = torch.diag(self.pred_adj_np)

        else:
            self.pred_adj_np = np.load('/home/lvxinyu/lib/scene-graph-benchmark/misc/conf_mat_vctree_train.npy')
            self.pred_adj_np[0, :] = 0.0
            self.pred_adj_np[:, 0] = 0.0
            self.pred_adj_np[0, 0] = 1.0
            self.pred_adj_np = self.pred_adj_np / (self.pred_adj_np.sum(-1)[:, None] + 1e-8)
            self.pred_adj_np = adj_laplacian(self.pred_adj_np)
            self.pred_adj_np = torch.from_numpy(self.pred_adj_np).float().cuda()
            self.pred_adj_np_diag = torch.diag(self.pred_adj_np)



        if self.use_logits_reweight:
            self.mycriterion_loss = MyCrossEntropyLoss()
        if self.use_contra_loss:
            if self.use_contra_bce:
                self.criterion_loss_contra_cb = TripletLoss_CB(self.use_contra_distance_loss_value)
            else:
                self.criterion_loss_contra = TripletLoss(self.use_contra_distance_loss_value)
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()

        self.iter = 0
    def loss_with_bias_level(self, relation_logits, rel_labels):
        num_rels = 51
        bias_level = int(relation_logits.size(-1) / num_rels)
        loss = 0
        # level_weight = np.ones(bias_level)
        level_weight = np.array([0.0, 1.0])
        for level_i in range(bias_level):
            level_i_start = level_i * num_rels
            level_i_end = (level_i + 1) * num_rels
            level_sample_mask = (rel_labels >= level_i_start) * (rel_labels < level_i_end)
            bg_sample_mask = (rel_labels == 0)
            level_bg_sample_mask = (level_sample_mask + bg_sample_mask) > 0
            if level_bg_sample_mask.size(0) > 0:
                rel_labels[level_sample_mask] = rel_labels[level_sample_mask] - level_i_start
                rel_labels_level = rel_labels[level_bg_sample_mask]
                relation_logits_level = relation_logits[level_bg_sample_mask, level_i_start:level_i_end]
                loss_temp = self.criterion_loss(relation_logits_level, rel_labels_level)
                loss = loss + level_weight[level_i] * loss_temp
        return loss

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, iteration):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """

        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        elif self.use_extra_loss:
            rel_dists = relation_logits
            refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits
        relation_logits = cat(rel_dists, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation_total = self.criterion_loss(relation_logits, rel_labels.long())  # [1214,51], [1214,1]
        if self.use_logits_reweight: # CDL
            # Extract predicate correlation from Biased Predicate Prediction and construct Predicate Lattice with predicate-predicate association
            # Distibution-based re-weighting
            predicate_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
                               14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0,
                               27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0,
                               40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0}
            rel_label_list = rel_labels.cpu().numpy().tolist()
            rel_label_list.sort()
            com_adj = torch.mul(self.pred_adj_np, torch.reciprocal(self.pred_adj_np_diag.reshape(51, 1).repeat(1, 51) + 0.0001))
            for key in rel_label_list:
                predicate_count[key] = predicate_count.get(key, 0) + 1
            dict_value_list = list(predicate_count.values())
            chushu = torch.reciprocal((torch.Tensor(dict_value_list)+ 1e-8).reshape(51, 1).repeat(1, 51).cuda())
            beichushu = (torch.Tensor(dict_value_list)+1e-8).reshape(51, 1).repeat(1, 51).cuda().t()
            category_statistics_ = torch.mul(beichushu, chushu)
            category_statistics_ = torch.clamp(category_statistics_, 0, 1000)
            # Adjust re-weighting process considering distribution and predicate correlations
            category_statistics1 = category_statistics_.pow(1.5)   # ni>nj  pj/pi<=0.9
            category_statistics2 = category_statistics_.pow(0.0)   # nj>=ni pj/pi<=0.9 or nj<ni pj/pi>0.9
            category_statistics3 = category_statistics_.pow(2.0)   # nj>=ni pj/pi>0.9
            category_statistics11 = torch.where(com_adj > 0.9, category_statistics2, category_statistics1) # ni>nj
            category_statistics22 = torch.where(com_adj > 0.9, category_statistics3, category_statistics2) # nj>=ni
            category_statistics_miti = torch.log(torch.where(category_statistics_ < 1 , category_statistics11, category_statistics22)+1)
            factor = category_statistics_miti[rel_labels]
            # CDL calculation
            loss_relation_total = self.mycriterion_loss(relation_logits, rel_labels.long(), factor)

        if self.use_contra_loss: # EDL
            # gather probabilities of positive predicates and hard-to-distinguish predicates in each sample
            relation_logits = F.log_softmax(relation_logits,dim=-1)
            pred_adj_np_index = torch.topk(self.pred_adj_np, self.candidate_number).indices# gather probabilities of hard-to-distinguish predicates in each sample based on predicate correlation
            negtive_index = pred_adj_np_index[rel_labels]
            negative = torch.gather(relation_logits,1,negtive_index.long())
            positive = torch.gather(relation_logits, 1, rel_labels.reshape(rel_labels.size(0),1).long()).repeat(1,self.candidate_number)
            # Balancing Factor
            predicate_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
                               14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0,
                               27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0,
                               40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0}
            rel_label_list = rel_labels.cpu().numpy().tolist()
            rel_label_list.sort()
            for key in rel_label_list:
                predicate_count[key] = predicate_count.get(key, 0) + 1
            dict_value_list = list(predicate_count.values())
            chushu = torch.reciprocal(torch.Tensor(dict_value_list).reshape(51, 1).repeat(1, 51).cuda())
            beichushu = torch.Tensor(dict_value_list).reshape(51, 1).repeat(1, 51).cuda().t()
            category_statistics = torch.mul(beichushu, chushu).pow(1.0)
            negative_weights = torch.gather(category_statistics[rel_labels], 1, negtive_index.long())
            weights = negative_weights
            # EDL calculation
            loss_relation_contrastive = self.criterion_loss_contra_cb(positive, negative, weights)
            # Add up EDL and CDL
            loss_relation_total += self.use_contra_distance_loss_cof * loss_relation_contrastive
            loss_relation = (loss_relation_total, loss_relation_contrastive)
        else:
            loss_relation = (loss_relation_total, loss_relation_total*0.0)

        if refine_obj_logits.requires_grad:
            loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        else:
            loss_refine_obj = torch.zeros(loss_relation[0].size()).float().to(loss_relation[0].get_device())

        ###############################################################              FPGL   End                     #######################################################################
        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets,
                                                  fg_bg_sample=self.attribute_sampling,
                                                  bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

    def conf_mat_loss(self, input, target, conf_mat):
        input = F.softmax(input, -1)
        # conf_mat = F.softmax(conf_mat, -1)
        conf_mat_pro = conf_mat / (conf_mat.sum(-1)[:, None] + 1e-8)
        input_conf = input @ conf_mat_pro.T
        loss = self.nllloss(torch.log(input_conf + 1e-8), target)
        return loss

def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cfg.MODEL.ROI_RELATION_HEAD.USE_EXTRA_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.USE_LOGITS_REWEIGHT,
        cfg.MODEL.ROI_RELATION_HEAD.USE_BCE_BATCH,
        cfg.MODEL.ROI_RELATION_HEAD.MITIGATION_FACTOR_HYPER,
        cfg.MODEL.ROI_RELATION_HEAD.USE_MITI,
        cfg.MODEL.ROI_RELATION_HEAD.MITI_TAIL,
        cfg.MODEL.ROI_RELATION_HEAD.USE_COM,
        cfg.MODEL.ROI_RELATION_HEAD.COM_ADJ,
        cfg.MODEL.ROI_RELATION_HEAD.USE_PCM,
        cfg.MODEL.ROI_RELATION_HEAD.COMPENSATION_FACTOR_HYPRT,
        cfg.MODEL.ROI_RELATION_HEAD.USE_CATEGORY_REWEIGHT,
        cfg.MODEL.ROI_RELATION_HEAD.USE_BCE,
        cfg.MODEL.ROI_RELATION_HEAD.PRED_WEIGHT_BETA,
        cfg.MODEL.ROI_RELATION_HEAD.USE_CDE,
        cfg.MODEL.ROI_RELATION_HEAD.USE_FOCAL,
        cfg.MODEL.ROI_RELATION_HEAD.FOCAL_GAMMA,
        cfg.MODEL.ROI_RELATION_HEAD.USE_CONTRA_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.USE_CONTRA_BCE,
        cfg.MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_VALUE,
        cfg.MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_COF,
        cfg.MODEL.ROI_RELATION_HEAD.CANDIDATE_NUMBER,
        cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR
    )
    return loss_evaluator
