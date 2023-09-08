# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Voting module: generate votes from XYZ and features of seed points.
Date: July, 2019
Author: Charles R. Qi and Or Litany
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.utils_epn.anchors as L
from models.blocks_epn import InvOutBlockEPN

class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim, config, kanchor=1): #, rot_semantic_cls=1, offset_semantic_cls=1, ):
        """ Votes generation from seed point features.
        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self.kanchor = kanchor
        self.quotient_factor = 5 if self.kanchor == 12 else 1
        self.rot_semantic_cls = config.rot_semantic_cls
        self.offset_semantic_cls = config.offset_semantic_cls
        
        self.rot_obj_pool = config.rot_obj_pool
        self.early_fuse_here = self.kanchor > 1 and config.early_fuse_obj and not config.share_fuse_obj
        self.early_fuse_ex = self.kanchor > 1 and config.early_fuse_obj and config.share_fuse_obj
        self.early_fuse = self.kanchor > 1 and config.early_fuse_obj
        self.effective_kanchor = 1 if self.early_fuse else self.kanchor
        self.att_obj_pooling = 'attn' in self.rot_obj_pool and not config.att_obj_permute #config.att_obj_pooling #if hasattr(config, 'att_pooling') else True
        self.att_obj_permute = 'attn' in self.rot_obj_pool and config.att_obj_permute

        self.neq_rot_cls = config.neq_rot_cls
        self.neq_kanchor = config.neq_kanchor
        if self.early_fuse_here:
            self.inv_layer = InvOutBlockEPN('inv_epn_obj', self.in_dim, config, obj_mode=True)
        if self.effective_kanchor == 1:
            self.in_dim = self.in_dim * self.kanchor if self.att_obj_permute else self.in_dim
            self.out_dim = self.in_dim
            self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
            self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
            if self.neq_rot_cls:
                self.conv3 = torch.nn.Conv1d(self.in_dim, (3*self.neq_kanchor + self.neq_kanchor + self.out_dim) * self.vote_factor, 1)
            else:
                self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + self.out_dim) * self.vote_factor, 1)
            self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
            self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        else:
            self.conv1 = torch.nn.Conv2d(self.in_dim, self.in_dim, 1)
            self.conv2 = torch.nn.Conv2d(self.in_dim, self.in_dim, 1)
            self.conv3 = torch.nn.Conv2d(self.in_dim, 
                (3*self.offset_semantic_cls + self.rot_semantic_cls + self.out_dim) * self.vote_factor, 1)
            self.bn1 = torch.nn.BatchNorm2d(self.in_dim)
            self.bn2 = torch.nn.BatchNorm2d(self.in_dim)

        if self.kanchor > 1:
            # rot_zs
            self.anchors = self.init_anchors(self.kanchor)
        elif self.neq_rot_cls and self.neq_kanchor > 1:
            self.anchors = self.init_anchors(self.neq_kanchor)

    def init_anchors(self, kanchor):
        # get so3 anchors (60x3x3 rotation matrices)
        if self.quotient_factor == 1:
            if kanchor < 10:    # SO(2)
                anchors = L.get_anchors(kanchor)
            else:   # SO(3)
                assert kanchor == 60, kanchor
                anchors = L.get_anchorsV()
        else:
            if kanchor < 10:    # SO(2)
                anchors = L.get_anchors(kanchor * self.quotient_factor)[:kanchor]
                quotient_anchors = L.get_anchors(self.quotient_factor)
                self.quotient_anchors = nn.Parameter(torch.tensor(quotient_anchors, dtype=torch.float32),
                            requires_grad=False)
            else:   # SO(3)
                assert kanchor == 12, kanchor
                assert self.quotient_factor == 5, self.quotient_factor
                anchors = L.get_anchorsV().reshape(12, 5, 3, 3)[:, 0]
                quotient_anchors = L.get_anchors(self.quotient_factor)
                self.quotient_anchors = nn.Parameter(torch.tensor(quotient_anchors, dtype=torch.float32),
                            requires_grad=False)

        return nn.Parameter(torch.tensor(anchors, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, seed_xyz, seed_features, point_semantic_classes=None, rot_cls=None):
        """ Forward pass.
        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            if self.kanchor == 1:
                seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
                point_semantic_classes: (batch_size, num_seed)
            else:
                seed_features: (batch_size, feature_dim, num_seed, kanchor) Pytorch tensor
                point_semantic_classes: (batch_size, num_seed) (kanchor marginalized)
            rot_cls: (num_seed, kanchor)
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed * self.vote_factor
        if self.early_fuse_ex:
            assert seed_features.ndim == 3, seed_features.shape
        if self.early_fuse_here:
            seed_features = seed_features.squeeze(0).permute(1,2,0)     # bcna->nac
            seed_features, rot_cls = self.inv_layer(seed_features)    # nc, na
            seed_features = seed_features.transpose(0,1).unsqueeze(0)   # 1cn
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, (3+out_dim)*vote_factor, num_seed, [kanchor])

        if point_semantic_classes is not None:
            # point_semantic_classes: (batch_size, num_seed, self.vote_factor, 1_from_semantic_cls, n_dim, self.kanchor)
            point_semantic_classes = point_semantic_classes[..., None, None, None, None]

        if self.effective_kanchor == 1:
            if self.neq_rot_cls:
                net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 4*self.neq_kanchor + self.out_dim)
                offset = net[:, :, :, 0:3*self.neq_kanchor]  # bn1(3a)
                rot_cls = net[:, :, :, 3*self.neq_kanchor:4*self.neq_kanchor]   # bn1a
                rot_idx = torch.max(rot_cls, dim=-1)[1]    # bn1
                offset = offset.reshape(*offset.shape[:2], self.neq_kanchor, 3) # bna3
                offset = torch.gather(offset, 2, rot_idx.unsqueeze(-1).expand(*offset.shape[:2], 1, 3))   # bn13
                rot_mats = self.anchors[rot_idx]    # bn133
                offset = torch.matmul(rot_mats, offset.unsqueeze(-1)).squeeze(-1)   #bn13
                rot_cls = rot_cls.permute(0,3,1,2).squeeze(-1)  #ban

                residual_features = net[:, :, :, 4*self.neq_kanchor:]  # (batch_size, num_seed, vote_factor, out_dim)
            else:
                net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 3 + self.out_dim)
                offset = net[:, :, :, 0:3]  # bn13
                residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
            if self.early_fuse and self.rot_obj_pool == 'attn_best':
                ### rotate offset
                rot_idx = torch.max(rot_cls, dim=1)[1]    # n
                rot_mats = self.anchors[rot_idx]    # n*3*3
                rot_mats = rot_mats[None,:,None]    # bn133
                offset = torch.matmul(rot_mats, offset.unsqueeze(-1)).squeeze(-1)   #bn13
            if self.early_fuse and 'attn' in self.rot_obj_pool:
                # na -> ban
                rot_cls = rot_cls.unsqueeze(0).transpose(1,2)
            vote_xyz = seed_xyz.unsqueeze(2) + offset
            vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

            vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
            vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
            vote_features = vote_features.transpose(2, 1).contiguous()
            
            # rot_cls = None
        else:
            ### (batch_size, (3*offset_semantic_cls + rot_semantic_cls + self.out_dim)*vote_factor, num_seed, kanchor)
            ### classification on the kanchor dim, and pick the features from the best
            net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 
                3*self.offset_semantic_cls + self.rot_semantic_cls + self.out_dim, self.kanchor)

            ### rot_cls: (batch_size, num_seed, self.vote_factor, self.rot_semantic_cls, kanchor)
            rot_cls = net[:,:,:, 3*self.offset_semantic_cls:3*self.offset_semantic_cls + self.rot_semantic_cls]
            if self.rot_semantic_cls > 1:
                # rot_cls = rot_cls.unsqueeze(3)
                rot_cls = rot_cls.reshape(*rot_cls.shape[:3], 1, self.rot_semantic_cls, self.kanchor)
                point_semantic_classes_ = point_semantic_classes.expand(-1,-1,-1,1,1,self.kanchor)
                rot_cls = torch.gather(rot_cls, -2, point_semantic_classes_).squeeze(-2) # (batch_size, num_seed, self.vote_factor, 1, self.kanchor)
            # (batch_size, num_seed, self.vote_factor, 1, 1)
            _, rot_cls_best_anchor = torch.max(rot_cls, -1, keepdim=True)
            # (batch_size, self.kanchor, num_vote)
            rot_cls = rot_cls.view(batch_size, num_vote, self.kanchor)
            rot_cls = rot_cls.transpose(2, 1) #.contiguous()

            ### offset: (batch_size, num_seed, self.vote_factor, 3*self.offset_semantic_cls, kanchor)
            offset = net[:,:,:, :3*self.offset_semantic_cls]
            if self.offset_semantic_cls > 1:
                offset = offset.reshape(*offset.shape[:3], 3, self.offset_semantic_cls, self.kanchor)
                # point_semantic_classes_: (batch_size, num_seed, self.vote_factor, self.offset_semantic_cls, 3, self.kanchor)
                point_semantic_classes_ = point_semantic_classes.expand(-1,-1,-1,3,1,self.kanchor)
                offset = torch.gather(offset, -2, point_semantic_classes_).squeeze(-2)   # (batch_size, num_seed, self.vote_factor, 3, self.kanchor)
            # (batch_size, num_seed, self.vote_factor, 3, 1)
            rot_cls_best_anchor_ = rot_cls_best_anchor.expand(-1,-1,-1, 3, 1)
            # offset: (batch_size, num_seed, self.vote_factor, 3, 1)
            offset = torch.gather(offset, -1, rot_cls_best_anchor_)
            # offset: (batch_size, num_seed, self.vote_factor, 3)
            rot_mats = self.anchors[rot_cls_best_anchor.flatten(2)]    # (batch_size, num_seed, self.vote_factor, 3, 3)
            offset = torch.matmul(rot_mats, offset).squeeze(-1)

            vote_xyz = seed_xyz.unsqueeze(2) + offset
            vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

            ### residual_features: (batch_size, num_seed, self.vote_factor, self.out_dim, kanchor)
            residual_features = net[:,:,:, 3*self.offset_semantic_cls + self.rot_semantic_cls:]
            rot_cls_best_anchor_ = rot_cls_best_anchor.expand(-1,-1,-1, self.out_dim, 1)
            # (batch_size, num_seed, self.vote_factor, self.out_dim)
            residual_features = torch.gather(residual_features, -1, rot_cls_best_anchor_).squeeze(-1)   

            # (batch_size, num_seed, self.vote_factor, feature_dim, kanchor)
            seed_features = seed_features.transpose(2, 1).unsqueeze(2)
            # (batch_size, num_seed, self.vote_factor, feature_dim)
            seed_features = torch.gather(seed_features, -1, rot_cls_best_anchor_).squeeze(-1)  

            vote_features = seed_features + residual_features
            vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
            vote_features = vote_features.transpose(2, 1).contiguous()

            # # offset_at_best_anchor: (batch_size, num_seed, self.vote_factor, 3*self.offset_semantic_cls)
            # if self.offset_semantic_cls == 1 and self.rot_semantic_cls == 1:
            #     rot_cls_best_anchor_ = rot_cls_best_anchor.reshape(-1, -1, -1, 1, self.rot_semantic_cls, 1).expand(
            #         -1, -1, -1, 3, self.rot_semantic_cls, 1)
            #     offset_at_best_anchor = torch.gather(offset, -1, rot_cls_best_anchor_).squeeze(-1) #(-1, -1, -1, 3, self.offset_semantic_cls)

            #     rot_cls_best_anchor_ = rot_cls_best_anchor.reshape(-1, -1, -1, 1, self.rot_semantic_cls, 1).expand(
            #         -1, -1, -1, self.out_dim, self.rot_semantic_cls, 1)
            #     residual_features = residual_features.reshape(-1,-1,-1, self.out_dim, 1, 1)
            #     # (batch_size, num_seed, self.vote_factor, self.out_dim, self.rot_semantic_cls)
            #     residual_features_at_best_anchor = torch.gather(residual_features, -1, rot_cls_best_anchor_).squeeze(-1)

            # elif self.offset_semantic_cls > 1 and self.rot_semantic_cls == 1:
            #     rot_cls_best_anchor_ = rot_cls_best_anchor.reshape(-1, -1, -1, 1, self.rot_semantic_cls, 1).expand(
            #         -1, -1, -1, 3, self.offset_semantic_cls, 1)
            #     offset_at_best_anchor = torch.gather(offset, -1, rot_cls_best_anchor_).squeeze(-1) #(-1, -1, -1, 3, self.offset_semantic_cls)

            #     rot_cls_best_anchor_ = rot_cls_best_anchor.reshape(-1, -1, -1, 1, self.rot_semantic_cls, 1).expand(
            #         -1, -1, -1, self.out_dim, self.rot_semantic_cls, 1)
            #     residual_features = residual_features.reshape(-1,-1,-1, self.out_dim, 1, 1)
            #     # (batch_size, num_seed, self.vote_factor, self.out_dim, self.rot_semantic_cls)
            #     residual_features_at_best_anchor = torch.gather(residual_features, -1, rot_cls_best_anchor_).squeeze(-1)

            # elif self.offset_semantic_cls == 1 and self.rot_semantic_cls > 1:
            #     raise NotImplementedError("Need semantic prediction to decide which rot anchor to use.")

            # elif self.offset_semantic_cls > 1 and self.rot_semantic_cls > 1:
            #     assert self.offset_semantic_cls == self.rot_semantic_cls, "{} {}".format(self.offset_semantic_cls, self.rot_semantic_cls)
            #     rot_cls_best_anchor_ = rot_cls_best_anchor.reshape(-1, -1, -1, 1, self.rot_semantic_cls, 1).expand(
            #         -1, -1, -1, 3, self.rot_semantic_cls, 1)
            #     offset_at_best_anchor = torch.gather(offset, -1, rot_cls_best_anchor_).squeeze(-1) #(-1, -1, -1, 3, self.offset_semantic_cls)
            #     assert point_semantic_classes is not None, "Need semantic prediction to decide which rot anchor to use."

            # else:
            #     raise NotImplementedError(f"Not recognized self.offset_semantic_cls={self.offset_semantic_cls}, self.rot_semantic_cls={self.rot_semantic_cls}")

        return vote_xyz, vote_features, rot_cls


if __name__ == '__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8, 1024, 3).cuda(), torch.rand(8, 256, 1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)