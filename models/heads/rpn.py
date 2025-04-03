import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from numpy.core._simd import targets


class RegionProposalNetwork(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=256,
            anchor_sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
    ):
        super(RegionProposalNetwork, self).__init__()

        self.anchor_generator = ops.AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )

        self.head = RPNHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0]
        )

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def forward(self, images, features, targets=None):
        features_list = [features[f] for f in features.keys() if f != 'fused']

        anchors = self.anchor_generator(images, features_list)

        objectness, pred_bbox_deltas = self.head(features_list)

        num_images = len(anchors)
        num_anchors_per_level = [o.shape[1] for o in objectness]
        objectness = torch.cat([o.flatten() for o in objectness], dim=0)

        pred_bbox_deltas = torch.cat(
            [b.view(num_images, -1, 4) for b in pred_bbox_deltas], dim=1
        ).reshape(-1, 4)

        loss_dict = {}
        if self.training and targets is not None:
            matched_idxs = []
            for i in range(num_images):
                matched_idxs_per_image = self.assign_targets_to_anchors(
                    anchors[i], targets[i]
                )
                matched_idxs.append(matched_idxs_per_image)

            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, matched_idxs
            )
            loss_dict = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }

        proposals = self.generate_proposals(
            anchors, objectness, pred_bbox_deltas, images.shape[-2:]
        )

        return proposals, loss_dict

    def assign_targets_to_anchors(self, anchors, targets):
        ious = ops.box_iou(anchors, targets)

        matched_vals, matched_idxs = ious.max(dim=1)

        matched_idxs[matched_vals < self.bg_iou_thresh] = -1  # 背景
        matched_idxs[matched_vals >= self.fg_iou_thresh] = 1  # 前景

        pos_idx = torch.where(matched_idxs == 1)[0]
        neg_idx = torch.where(matched_idxs == -1)[0]

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(pos_idx.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(neg_idx.numel(), num_neg)

        perm1 = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:num_pos]
        perm2 = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]

        pos_idx = pos_idx[perm1]
        neg_idx = neg_idx[perm2]

        sampled_idxs = torch.cat([pos_idx, neg_idx])
        sampled_matched_idxs = matched_idxs[sampled_idxs]

        return sampled_matched_idxs

    def compute_loss(self, objectness, pred_bbox_deltas, matched_idxs, sampled_idxs=None):
        objectness_targets = torch.cat([
            torch.ones_like(sampled_idxs[sampled_idxs >= 0]),
            torch.zeros_like(sampled_idxs[sampled_idxs < 0])
        ], dim=0)

        loss_objectness = F.binary_cross_entropy_with_logits(
            objectness, objectness_targets
        )

        loss_rpn_box_reg = F.smooth_l1_loss(
            pred_bbox_deltas[matched_idxs >= 0],
            targets[matched_idxs[matched_idxs >= 0]],
            reduction="sum"
        ) / max(1, torch.sum(matched_idxs >= 0).item())

        return loss_objectness, loss_rpn_box_reg

    def generate_proposals(self, anchors, objectness, pred_bbox_deltas, image_size):
        proposals = []

        for i, (anchors_per_image, objectness_per_image, pred_bbox_deltas_per_image) in enumerate(
                zip(anchors, objectness, pred_bbox_deltas)
        ):
            pre_nms_top_n = self.pre_nms_top_n['training'] if self.training else self.pre_nms_top_n['testing']
            post_nms_top_n = self.post_nms_top_n['training'] if self.training else self.post_nms_top_n['testing']

            proposals_per_image = self.apply_deltas_to_anchors(
                pred_bbox_deltas_per_image, anchors_per_image
            )

            proposals_per_image = self.clip_proposals_to_image(
                proposals_per_image, image_size
            )

            keep = self.remove_small_boxes(proposals_per_image, self.min_size)

            proposals_per_image = proposals_per_image[keep]
            objectness_per_image = objectness_per_image[keep]

            top_n = min(pre_nms_top_n, objectness_per_image.shape[0])
            _, topk_idx = objectness_per_image.topk(top_n, sorted=True)

            proposals_per_image = proposals_per_image[topk_idx]
            objectness_per_image = objectness_per_image[topk_idx]

            keep = ops.nms(
                proposals_per_image,
                objectness_per_image,
                self.nms_thresh
            )

            keep = keep[:post_nms_top_n]
            proposals_per_image = proposals_per_image[keep]

            proposals.append(proposals_per_image)

        return proposals

    def apply_deltas_to_anchors(self, deltas, anchors):
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        ctr_x = ctr_x + dx * widths
        ctr_y = ctr_y + dy * heights
        widths = widths * torch.exp(dw)
        heights = heights * torch.exp(dh)

        x1 = ctr_x - 0.5 * widths
        y1 = ctr_y - 0.5 * heights
        x2 = ctr_x + 0.5 * widths
        y2 = ctr_y + 0.5 * heights

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes

    def clip_proposals_to_image(self, proposals, image_size):
        height, width = image_size
        proposals[:, 0].clamp_(min=0, max=width)
        proposals[:, 1].clamp_(min=0, max=height)
        proposals[:, 2].clamp_(min=0, max=width)
        proposals[:, 3].clamp_(min=0, max=height)
        return proposals

    def remove_small_boxes(self, boxes, min_size):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= min_size) & (heights >= min_size)
        return keep


class RPNHead(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_anchors):
        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(hidden_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=1)

    def forward(self, features):
        objectness = []
        bbox_pred = []

        for feature in features:
            t = F.relu(self.conv(feature))
            objectness.append(self.objectness(t))
            bbox_pred.append(self.bbox_pred(t))

        return objectness, bbox_pred
