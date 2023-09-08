#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import *
from models.blocks_all import block_decider
from models.blocks_epn import UnaryBlockEPN
from models.losses import *
import numpy as np
import torch.nn as nn
import torch


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        self.feature_dim = config.first_features_dim
        self.equiv = 'epn' in config.architecture[0]
        self.multi_rot_head = self.equiv and config.architecture[-1] == 'unary_epn'
        print('self.multi_rot_head={}'.format(self.multi_rot_head))
        self.early_fuse = self.equiv and config.architecture[-1] == 'inv_epn' #not self.multi_rot_head 
        self.effective_kanchor = 1 if self.early_fuse else config.kanchor
        self.obj_inv_f = self.early_fuse and config.early_fuse_obj and config.share_fuse_obj
        self.att_permute = self.early_fuse and 'attn' in config.rot_head_pool and config.att_permute
        # self.rot_head_attn = config.rot_head_attn
        self.rot_head_pool = config.rot_head_pool
        self.dual_feature = self.equiv and config.dual_feature
        self.ctrness_w_track = config.ctrness_w_track
        self.kanchor = config.kanchor
        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            elif 'lift' in block:
                pass
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        decoders_blocks = config.architecture[start_i:-1] if self.early_fuse else config.architecture[start_i:]
        # Loop over consecutive blocks
        for block_i, block in enumerate(decoders_blocks):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        if self.dual_feature:
            self.head_mlp_max = UnaryBlock(out_dim, self.feature_dim, False, 0)

        if self.equiv:
            self.head_mlp = UnaryBlockEPN(out_dim, self.feature_dim, False, 0)
        else:
            self.head_mlp = UnaryBlock(out_dim, self.feature_dim, False, 0)

        # otherwise, out_dim == self.feature_dim == 256
        if self.effective_kanchor == 1:
            # self.head_mlp = UnaryBlock(out_dim, self.feature_dim, False, 0)
            if self.early_fuse:
                self.inv_layer = block_decider(config.architecture[-1],
                                                r,
                                                self.feature_dim,
                                                self.feature_dim,
                                                layer,
                                                config)
                if self.att_permute:
                    self.feature_dim = self.feature_dim * self.kanchor
            self.head_var = UnaryBlock(self.feature_dim, out_dim + config.free_dim, False, 0)
            self.head_softmax = UnaryBlock(self.feature_dim, self.C, False, 0)
            self.head_center = UnaryBlock(self.feature_dim, 1, False, 0, False)
        else:
        # if self.multi_rot_head:
            # self.head_mlp = UnaryBlockEPN(out_dim, self.feature_dim, False, 0)
            self.head_var = UnaryBlockEPN(self.feature_dim, out_dim + config.free_dim, False, 0)
            self.head_softmax = UnaryBlockEPN(self.feature_dim, self.C, False, 0)
            self.head_center = UnaryBlockEPN(self.feature_dim, 1, False, 0, False)
            if self.rot_head_pool in ['attn_soft', 'attn_best']:
                self.head_rot_weight = UnaryBlockEPN(self.feature_dim, 1, False, 0, True)

        self.pre_train = config.pre_train
        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.center_loss = 0
        self.instance_loss = torch.tensor(0)
        self.variance_loss = torch.tensor(0)
        self.instance_half_loss = torch.tensor(0)
        self.reg_loss = 0
        self.variance_l2 = torch.tensor(0)
        self.l1 = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, batch):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=-1)
            x = block_op(x, batch)

        f = self.head_mlp(x, batch)
        f_obj = f
        attention = None
        if self.early_fuse:
            f, attention = self.inv_layer(f)
            if self.obj_inv_f:
                f_obj = f

        # Head of network
        c = self.head_center(f, batch)
        v = self.head_var(f, batch)
        s = self.head_softmax(f, batch)

        if self.multi_rot_head:
            if self.rot_head_pool in ['attn_soft', 'attn_best']:
                rw = self.head_rot_weight(f, batch)
                attention = rw.squeeze(-1)
                if self.rot_head_pool == 'attn_soft':
                    rw = F.softmax(rw, 1)
                    # f = (f * rw).sum(1)
                    c = (c * rw).sum(1)
                    v = (v * rw).sum(1)
                    s = (s * rw).sum(1)
                else:
                    rw = torch.max(rw, 1)[1].squeeze()  #np,na=1,nc=1
                    c = c[torch.arange(c.shape[0]), rw] # np, nc
                    v = v[torch.arange(v.shape[0]), rw] # np, nc
                    s = s[torch.arange(s.shape[0]), rw] # np, nc
            elif self.rot_head_pool == 'mean':
                # f = f.mean(1)
                c = c.mean(1)
                v = v.mean(1)
                s = s.mean(1)
            elif self.rot_head_pool == 'max':
                ### max does not make much sense here? since they are all predictions
                # f = f.mean(1)
                c = c.amax(1)
                v = v.amax(1)
                s = s.amax(1)
            else:
                raise NotImplementedError(f'self.rot_head_pool={self.rot_head_pool} not recognized') 

        c = self.sigmoid(c)
        v = F.relu(v)

        #return x, c, v, f
        # attention: na
        # s, c: nc
        # f_out: nac or nc
        if f_obj.ndim == f.ndim:
            f_obj = None
        return s, c, f, f_obj, attention
