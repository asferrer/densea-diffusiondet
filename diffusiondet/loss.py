# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .fast_opt import flag as fast_flag
from .util import box_ops
from .util.misc import get_world_size, is_dist_avail_and_initialized
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.fast_loss = fast_flag("DENSEA_FAST_LOSS")  # no CPU/GPU synchronizations, same result
        if self.use_fed_loss:
            self.fed_loss_num_classes = min(50, self.num_classes)
            from detectron2.data.detection_utils import get_fed_loss_cls_weights
            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  # noqa
            fed_loss_cls_weights = cls_weight_fun()
            assert (
                    len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size = len(targets)

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if self.use_focal or self.use_fed_loss:
            num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            gt_classes = torch.argmax(target_classes_onehot, dim=-1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            src_logits = src_logits.flatten(0, 1)
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            if self.use_focal:
                if self.fast_loss:
                    # same as the `elif` below, without synchronizing: if there is any NaN/Inf the whole tensor is clamped
                    bad = (~torch.isfinite(src_logits)).any()
                    src_logits = torch.where(bad, torch.clamp(src_logits, min=-15, max=15), src_logits)
                elif torch.isnan(src_logits).any() or torch.isinf(src_logits).any():
                    src_logits = torch.clamp(src_logits, min=-15, max=15)
                cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            if self.use_fed_loss:
                K = self.num_classes
                N = src_logits.shape[0]
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights,
                )
                fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()

                loss_ce = torch.sum(cls_loss * weight) / num_boxes
            else:
                loss_ce = torch.sum(cls_loss) / num_boxes

            losses = {'loss_ce': loss_ce}
        else:
            raise NotImplementedError

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if self.fast_loss and not is_dist_avail_and_initialized():
            num_boxes = float(max(num_boxes, 1))  # same value as the float32 clamp, without a round trip to the GPU
        else:
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1, use_focal: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.ota_k = cfg.MODEL.DiffusionDet.OTA_K
        self.fast_loss = fast_flag("DENSEA_FAST_LOSS")
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]
            # 1) Compute probabilities in FP32 for classification cost
            if self.use_focal or self.use_fed_loss:
                logits_fp32 = outputs['pred_logits'].float()
                probs = logits_fp32.sigmoid()  # FP32
            else:
                probs = outputs['pred_logits'].softmax(-1)

            pred_boxes = outputs['pred_boxes']  # [bs, Q, 4]
            all_indices = []
            all_matched_ids = []

            for b in range(bs):
                bz_probs = probs[b]  # [Q, C]
                bz_boxes = pred_boxes[b]  # [Q, 4]
                tgt = targets[b]
                tgt_labels = tgt['labels']          # [num_gt]
                tgt_boxes_xyxy = tgt['boxes_xyxy'] # [num_gt, 4]
                num_gt = len(tgt_labels)

                # Empty target handling
                if num_gt == 0:
                    valid = torch.zeros(bz_probs.shape[0], dtype=torch.bool, device=bz_probs.device)
                    all_indices.append((valid, torch.arange(0, 0, device=bz_probs.device)))
                    all_matched_ids.append(torch.arange(0, 0, device=bz_probs.device))
                    continue

                # 2) Pairwise IoU
                pair_wise_ious = ops.box_iou(bz_boxes, tgt_boxes_xyxy)

                # 3) Classification cost
                eps = 1e-6  # representable in FP16
                if self.use_focal:
                    alpha, gamma = self.focal_loss_alpha, self.focal_loss_gamma
                    neg_cost = (1 - alpha) * (bz_probs ** gamma) * (-(1 - bz_probs + eps).log())
                    pos_cost = alpha * ((1 - bz_probs) ** gamma) * (-(bz_probs + eps).log())
                    cost_class = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]
                elif self.use_fed_loss:
                    neg_cost = (-(1 - bz_probs + eps).log())
                    pos_cost = (-(bz_probs + eps).log())
                    cost_class = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]
                else:
                    cost_class = -bz_probs[:, tgt_labels]

                # 4) BBox L1 cost
                img_size = tgt['image_size_xyxy']
                tgt_size = tgt['image_size_xyxy_tgt']
                cost_bbox = torch.cdist(
                    bz_boxes / img_size,
                    tgt_boxes_xyxy / tgt_size,
                    p=1
                )

                # 5) GIoU cost
                cost_giou = -generalized_box_iou(bz_boxes, tgt_boxes_xyxy)

                # 6) Final cost matrix
                cost_matrix = (
                    self.cost_class * cost_class
                    + self.cost_bbox * cost_bbox
                    + self.cost_giou * cost_giou
                )

                # 7) In-box mask
                fg_mask, center_mask = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes),
                    box_xyxy_to_cxcywh(tgt_boxes_xyxy),
                    expanded_strides=32
                )
                if self.fast_loss:
                    cost_matrix = torch.where(fg_mask.unsqueeze(1), cost_matrix, cost_matrix + 10000.0)
                else:
                    cost_matrix[~fg_mask] += 10000.0

                # 8) Dynamic k matching
                indices_b, matched_ids_b = self.dynamic_k_matching(
                    cost_matrix, pair_wise_ious, num_gt
                )
                all_indices.append(indices_b)
                all_matched_ids.append(matched_ids_b)

        return all_indices, all_matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        if self.fast_loss:
            return self.dynamic_k_matching_fast(cost, pair_wise_ious, num_gt)
        return self.dynamic_k_matching_ref(cost, pair_wise_ious, num_gt)

    @staticmethod
    def _keep_min_cost(matching_matrix, cost, rows):
        """In the rows `rows`, keep only the lowest-cost column (no mask indexing => no synchronization)."""
        argmin = torch.min(cost, dim=1)[1]
        onehot = F.one_hot(argmin, cost.shape[1]).to(matching_matrix.dtype)
        return torch.where(rows.unsqueeze(1), onehot, matching_matrix)

    def dynamic_k_matching_fast(self, cost, pair_wise_ious, num_gt):
        """Same assignments as dynamic_k_matching_ref with far fewer CPU/GPU synchronizations.
        The tie-breaking (topk/argmin/min by index) and the `anchor_matching_gt > 1` mask,
        computed only once, are preserved as in the original version."""
        matching_matrix = torch.zeros_like(cost)
        topk_ious, _ = torch.topk(pair_wise_ious, self.ota_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1).tolist()  # a single synchronization
        for gt_idx, k in enumerate(dynamic_ks):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=k, largest=False)
            matching_matrix[pos_idx, gt_idx] = 1.0

        multi = matching_matrix.sum(1) > 1
        matching_matrix = self._keep_min_cost(matching_matrix, cost, multi)

        cols = torch.arange(num_gt, device=cost.device)
        while (matching_matrix.sum(0) == 0).any():
            matched_query_id = matching_matrix.sum(1) > 0
            cost = torch.where(matched_query_id.unsqueeze(1), cost + 100000.0, cost)
            unmatched = matching_matrix.sum(0) == 0
            pos = torch.argmin(cost, dim=0)
            cur = matching_matrix[pos, cols]
            matching_matrix[pos, cols] = torch.where(unmatched, torch.ones_like(cur), cur)
            rows = multi & (matching_matrix.sum(1) > 1).any()
            matching_matrix = self._keep_min_cost(matching_matrix, cost, rows)

        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        cost = torch.where(matching_matrix == 0, cost + float('inf'), cost)
        matched_query_id = torch.min(cost, dim=0)[1]
        return (selected_query, gt_indices), matched_query_id

    def dynamic_k_matching_ref(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id
