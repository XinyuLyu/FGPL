import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps

from abc import ABC, abstractmethod

import copy
class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict

    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass

    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '  R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
            pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode == 'phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)

        return local_container



class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += 'ngR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] + 1))
        nogc_pred_scores = rel_scores[nogc_score_inds[:, 0], nogc_score_inds[:, 1] + 1]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
            nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode == 'phrdet',
        )

        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)


"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""


class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += ' zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""


class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + '_accuracy_count'][k])
            result_str += '  A @ %d: %.4f; ' % (k, a_hit / a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
            # for sgcls and predcls
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))


class SSGConfMat(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel_category, ind_to_predicates):
        super(SSGConfMat, self).__init__(result_dict)
        self.num_rel_category = num_rel_category
        self.ind_to_predicates = ind_to_predicates

    def register_container(self, mode):
        self.result_dict['predicate_confusion_matrix_soft'] = np.zeros([self.num_rel_category, self.num_rel_category],
                                                                       dtype='float32')

    def generate_print_string(self, mode):
        result_str = 'SGG confusion matrix has calculated! \n'
        return result_str

    def prepare_gtpair_soft(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = np.where(pred_pair_idx[:, None] == gt_pair_idx[None, :])

    def calculate_confusion_matrix_soft(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)
        pred_inds = self.pred_pair_in_gt[0]
        gt_inds = self.pred_pair_in_gt[1]
        # match the subject and object

        if mode == 'predcls':
            for i in range(len(pred_inds)):
                pred_ind = pred_inds[i]
                gt_ind = gt_inds[i]
                pred_pred_i = pred_rels[pred_ind][2]
                gt_pred_i = gt_rels[gt_ind][2]
                scores = pred_scores[pred_ind]
                if pred_pred_i < self.result_dict['predicate_confusion_matrix_soft'].shape[1] and \
                        gt_pred_i < self.result_dict['predicate_confusion_matrix_soft'].shape[0]:
                    self.result_dict['predicate_confusion_matrix_soft'][gt_pred_i][pred_pred_i] = \
                        self.result_dict['predicate_confusion_matrix_soft'][gt_pred_i][pred_pred_i] + scores


class SGConfMat(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel_category, ind_to_predicates):
        super(SGConfMat, self).__init__(result_dict)
        self.num_rel_category = num_rel_category
        self.ind_to_predicates = ind_to_predicates

    def register_container(self, mode):
        self.result_dict['predicate_confusion_matrix'] = np.zeros([self.num_rel_category, self.num_rel_category],
                                                                  dtype='float32')
        self.result_dict['predicate_confusion_matrix_soft'] = np.zeros([self.num_rel_category, self.num_rel_category],
                                                                       dtype='float32')

    def generate_print_string(self, mode):
        result_str = 'SGG confusion matrix has calculated! \n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = np.where(pred_pair_idx[:, None] == gt_pair_idx[None, :])

    def calculate_confusion_matrix(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)
        pred_inds = self.pred_pair_in_gt[0]
        gt_inds = self.pred_pair_in_gt[1]
        # match the subject and object

        if mode == 'predcls':
            for i in range(len(pred_inds)):
                pred_ind = pred_inds[i]
                gt_ind = gt_inds[i]
                pred_pred_i = pred_rels[pred_ind][2]
                gt_pred_i = gt_rels[gt_ind][2]
                if pred_pred_i < self.result_dict['predicate_confusion_matrix'].shape[1] and \
                        gt_pred_i < self.result_dict['predicate_confusion_matrix'].shape[0]:
                    self.result_dict['predicate_confusion_matrix'][gt_pred_i][pred_pred_i] = \
                        self.result_dict['predicate_confusion_matrix'][gt_pred_i][pred_pred_i] + 1


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""


class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]  # remove __background__

    def register_container(self, mode):
        # self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)],
                                                           50: [[] for i in range(self.num_rel)],
                                                           100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += ' mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                if local_label < len(recall_count):
                    recall_count[int(local_label)] += 1
                    recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx + 1])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return


"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""


class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += ' aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(
                self.result_dict[mode + '_recall_hit'][k][0]) / float(
                self.result_dict[mode + '_recall_count'][k][0] + 1e-10)

        return

def cal_vagueness(confusion_matrix):
    conf_mat = confusion_matrix[1:,1:]
    diag_ind = []
    for i in range(conf_mat.shape[0]):
        diag_ind.append([i,i])
    conf_mat = conf_mat / (conf_mat.sum(-1)[:,None]+1e-8)
    diag_ind = np.array(diag_ind)
    conf_mat_diag = copy.deepcopy(np.diagonal(conf_mat))
    print(np.max(conf_mat/(conf_mat_diag+1e-2)))
    print(np.min(conf_mat/(conf_mat_diag+1e-2)))
    conf_mat[diag_ind[:,0], diag_ind[:,1]] = -1.0

    conf_mat_sort = 0.0 - np.sort((0.0 - conf_mat), -1)
    top_k = [1,3,5,10,20, 30]
    vag_k = {}
    for top_i in top_k:
        vag_all = conf_mat_diag[:,None] - conf_mat_sort[:,:top_i]
        vag_all = vag_all * (vag_all >= 0.0)
        vag_k['top_'+str(top_i)] = vag_all.mean(-1).mean(-1)
    return vag_k
class SGDp(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGDp, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict['dp'] = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 30: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict['dp'].items():
            result_str += ' aR @ %s: %.4f; ' % (str('dp_'+str(k)), float(v))
        result_str += ' for mode=%s, type=DP.' % mode
        result_str += '\n'
        return result_str

    def calculate_dp(self, mode):
        vg = cal_vagueness(self.result_dict['predicate_confusion_matrix'])
        for k, v in self.result_dict['dp'].items():
            self.result_dict['dp'][k] = float(vg['top_'+str(k)])

        return

class SGGroup_Mean_Recall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGGroup_Mean_Recall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict['Group_Mean_Recall'] = {'head': 0.0, 'body': 0.0, 'tail': 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict['Group_Mean_Recall'].items():
            result_str += ' aR @ %s: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Group_Mean Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_group_recall(self, mode):
        sorted_logits = []
        list1 = ['on', 'has', 'in', 'of', 'wearing', 'near', 'with', 'above', 'holding', 'behind', 'under',
                 'sitting on', 'wears', 'standing on', 'in front of', 'attached to', 'at', 'hanging from', 'over',
                 'for', 'riding', 'carrying', 'eating', 'walking on', 'playing', 'covering', 'laying on', 'along',
                 'watching', 'and', 'between', 'belonging to', 'painted on', 'against', 'looking at', 'from',
                 'parked on', 'to', 'made of', 'covered in', 'mounted on', 'says', 'part of', 'across', 'flying in',
                 'using', 'on back of', 'lying on', 'growing on', 'walking in']
        input1 = [
            "(above:0.1970) (across:0.2381) (against:0.2661) (along:0.3685) (and:0.1612) (at:0.5716) (attached "
            "to:0.0826) (behind:0.6245) (belonging to:0.1060) (between:0.1481) (carrying:0.6129) (covered in:0.5943) "
            "(covering:0.5162) (eating:0.6509) (flying in:0.0000) (for:0.2041) (from:0.1915) (growing on:0.2263) ("
            "hanging from:0.3143) (has:0.6779) (holding:0.6334) (in:0.3942) (in front of:0.3007) (laying on:0.3033) ("
            "looking at:0.2289) (lying on:0.1917) (made of:0.4062) (mounted on:0.1236) (near:0.2367) (of:0.6522) ("
            "on:0.5343) (on back of:0.0780) (over:0.2167) (painted on:0.1216) (parked on:0.8387) (part of:0.0041) ("
            "playing:0.1705) (riding:0.8408) (says:0.1667) (sitting on:0.4336) (standing on:0.3071) (to:0.4153) ("
            "under:0.3804) (using:0.4080) (walking in:0.1495) (walking on:0.6643) (watching:0.4962) (wearing:0.9288) "
            "(wears:0.0182) (with:0.2886) "]
        labels = []
        logits = []
        list = input1[0].split(')')
        for i in range(len(list) - 1):
            logits.append(list[i].split(':')[1])
            labels.append(list[i].split(':')[0].split('(')[1])
        logits = self.result_dict['predcls_mean_recall_list'][50]
        for i in list1:
            sorted_logits.append(float(logits[labels.index(i)]))
        head = sum(sorted_logits[:16]) / 16
        common = sum(sorted_logits[16:33]) / (33 - 16)
        tail = sum(sorted_logits[33:]) / (50 - 33)
        self.result_dict['Group_Mean_Recall']['head'] = head
        self.result_dict['Group_Mean_Recall']['body'] = common
        self.result_dict['Group_Mean_Recall']['tail'] = tail

        return


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                          gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
