# Common libs
import signal
import argparse

# Dataset
from datasets.SemanticKitti import *
from models.architectures import KPFCNN
from utils.config import Config
from utils.trainer import ModelTrainer

from models.mpanet import MPAnet
from models.prototype import PrototypeNet
from models.mpanet_binary import MPAnetBinary

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import logging

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)#, init_method='tcp://127.0.0.1:12344')

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#
### equivariant version
eq_architecture = ['lift_epn',
                'simple_epn',
                'resnetb_epn',
                'resnetb_strided_epn',
                'resnetb_epn',
                'resnetb_epn',
                'resnetb_strided_epn',
                'resnetb_epn',
                'resnetb_epn',
                'resnetb_strided_epn',
                'resnetb_epn',
                'resnetb_epn',
                'resnetb_strided_epn',
                'resnetb_epn',
                'resnetb_epn',
                # 'resnetb_epn', #
                # 'resnetb_epn', #
                'resnetb_strided_epn',
                'resnetb_epn',
                # 'resnetb_epn', #
                'nearest_upsample',
                'unary_epn',
                'nearest_upsample',
                'unary_epn',
                'nearest_upsample',
                'unary_epn',
                'nearest_upsample',
                'unary_epn',
                'nearest_upsample',
                'unary_epn',
                # 'inv_epn', # comment out inv_epn to enable late fusion
                ]
class SemanticKittiConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SemanticKitti'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 16

    #########################
    # Architecture definition
    #########################

    # Define layers
    '''
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']
    '''
    ### non-equivariant version
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    # Should set in_radius 50 times bigger than first_subsampling_dl (In KPConv: first_subsampling_dl = 0.06, in_radius = 4.0)
    in_radius = 6.0 # in 4D-PLS
    #in_radius = 4.0
    val_radius = 51.0
    n_frames = 4
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    #batch_num = 4  # in 4D-PLS Code
    batch_num = 8 # in KPConv and in 4D-PLS uploaded model
    val_batch_num = 1

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    #first_subsampling_dl = 0.06 * 2 # in 4D-PLS Code
    first_subsampling_dl = 0.06 # in KPConv and in 4D-PLS uploaded model

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 256
    #in_features_dim = 3
    in_features_dim = 2
    #free_dim = 3
    free_dim = 4

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 800

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 200) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 200

    # Number of epoch between each checkpoint
    checkpoint_gap = 400

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    # class proportion for R=10.0 and dl=0.08 (first is unlabeled)
    # 19.1 48.9 0.5  1.1  5.6  3.6  0.7  0.6  0.9 193.2 17.7 127.4 6.7 132.3 68.4 283.8 7.0 78.5 3.3 0.8
    #
    #

    # sqrt(Inverse of proportion * 100)
    # class_w = [1.430, 14.142, 9.535, 4.226, 5.270, 11.952, 12.910, 10.541, 0.719,
    #            2.377, 0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.505, 11.180]

    # sqrt(Inverse of proportion * 100)  capped (0.5 < X < 5)
    # class_w = [1.430, 5.000, 5.000, 4.226, 5.000, 5.000, 5.000, 5.000, 0.719, 2.377,
    #            0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.000, 5.000]

    # Do we nee to save convergence
    saving = True
    saving_path = None

    # Only train class and center head
    pre_train = False

    ################ EPN config
    # reinit_var = False

    equivariant_mode = False
    epn_kernel = False
    kanchor = 1
    att_pooling = False
    att_permute = False
    dual_feature = False
    ctrness_w_track = False
    equiv_mode_kp = False
    non_sep_conv = False
    rot_by_permute = False
    quotient_factor = 1
    # rot_head_attn = False
    rot_head_pool = 'mean'
    rot_obj_pool = 'attn_best'
    att_obj_permute = False
    early_fuse_obj = False
    share_fuse_obj = False
    ignore_steer_constraint = False
    gather_by_idxing = False
    rot_semantic_cls = 1
    offset_semantic_cls = 1
    train_val = False
    neq_rot_cls = False
    neq_kanchor = 1
    nuscene = False

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

# if __name__ == '__main__':
def main(gpu_id, gpus, nodes, node_id, saving_path, args):

    world_size = gpus * nodes
    rank = gpus * node_id + gpu_id
    distributed = world_size > 1
    if distributed:
        ddp_setup(rank, world_size)
    main_proc = gpu_id == 0
    main_proc_glob = rank == 0

    # print(torch.__version__)

    ############################
    # Initialize the environment
    ############################

    # # Set which gpu is going to be used
    # GPU_ID = '0'
    # if torch.cuda.device_count() > 1:
    #     GPU_ID = '0, 1'

    # # Set GPU visible device
    # os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############
    # Specify args.log if you want to start training from a previous snapshot (None for new training)
    previous_training_path = args.log   #'Log_2023-02-09_20-28-07' # None # pre_train
    equivariant_mode = args.eq          # Only effective when previous_training_path is None
    print('previous_training_path: {}'.format(previous_training_path))
    second_stage = previous_training_path is not None and not args.recover

    # # Choose index of checkpoint to start from. If None, uses the latest chkp
    # chkp_idx = None
    if previous_training_path:
        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        # chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if args.chkp is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = 'chkp_{:04d}.tar'.format(args.chkp)
            # chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(chkp_path, chosen_chkp)
    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############


    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration 
    config = SemanticKittiConfig()
    if previous_training_path:
        path = os.path.join('results', previous_training_path)
        config.load(path)
        config.saving_path = None
        #config.saving_path = path
        config.checkpoint_gap = 100
        ### Make sure the loaded checkpoint matches the intended options.
        ### (No practical effect, just to make sure you know what you are training. )
        assert config.equivariant_mode == args.eq, 'Wrong equivariant_mode in the checkpoint.'
        assert config.first_features_dim == args.fdim, 'Wrong first_features_dim in the checkpoint.'
        assert config.n_frames == args.nframe, 'Wrong n_frames in the checkpoint.'
        if args.eq:
            assert config.kanchor == args.kanchor, 'Wrong kanchor in the checkpoint.'
        else:
            assert config.neq_kanchor == args.kanchor, 'Wrong kanchor in the checkpoint.'
    else:
        # config.max_in_points = 10000
        # config.max_val_points = 10000
        config.first_features_dim = args.fdim #64 #43 # 128 # 85
        config.n_frames = args.nframe
        config.equivariant_mode = equivariant_mode
        config.train_val = args.train_val
        config.input_threads = args.input_threads
        config.nuscene = args.nuscene
        # config.learning_rate = 2e-2
        # config.epoch_steps = 250
        if distributed:
            assert 8 % world_size == 0, world_size
            config.batch_num = int(8 / world_size)
            config.validation_size = int(200 / world_size)
            config.input_threads = int(config.input_threads / world_size)
        if config.equivariant_mode:
            config.architecture = eq_architecture
            if args.head_early:
                config.architecture.append('inv_epn')
            config.early_fuse_obj = args.early_fuse_obj
            config.rot_head_pool = args.rot_head_pool
            config.rot_obj_pool = args.rot_obj_pool
            config.att_permute = args.att_permute
            config.att_obj_permute = args.att_obj_permute
            config.rot_semantic_cls = args.rot_semantic_cls
            config.offset_semantic_cls = args.offset_semantic_cls
            config.share_fuse_obj = args.share_fuse_obj
            if config.share_fuse_obj:
                config.att_obj_permute = config.att_permute
                config.rot_obj_pool = config.rot_head_pool

            config.kanchor = args.kanchor #4
            if config.kanchor == 4:
                config.num_kernel_points = 19    # 15
            config.rot_by_permute = True # False #True
            # config.rot_head_pool = 'max'
            config.equiv_mode_kp = True # experiment its effect on equivariant models
            config.fixed_kernel_points = 'verticals'
            config.non_sep_conv = True
            config.gather_by_idxing = True
            # config.rot_semantic_cls = 1 #19 #1
            # config.offset_semantic_cls = 1 #19 #1
        else:
            if args.kanchor > 1:
                config.neq_rot_cls = True
                config.neq_kanchor = args.kanchor

    # config.max_epoch = 300 if previous_training_path else 800 #800#300 # pre_train
    config.max_epoch = 1100 if second_stage else 800 #800#300 # pre_train
    # config.lr_decays = {i: 0.1 ** (1 / 200) for i in range(1, config.max_epoch)}
    config.lr_decays = {i: 0.1 ** (1 / 200) for i in range(1, 800)}
    #config.learning_rate = 1e-1
    # config.learning_rate = 1e-2
    #config.learning_rate = 1e-3
    config.pre_train = not second_stage #True
    # config.pre_train = False
    config.reinit_var = False
    #config.n_frames = 2
    # config.n_frames = 4
    config.n_test_frames = 1
    config.stride = 1
    #config.sampling = 'objectness'
    config.sampling = 'importance'
    #config.sampling = 'None'
    config.decay_sampling = 'None'
    #config.freeze = True
    config.freeze = not config.pre_train #False
    #config.lr_exponential_decrease = False
    config.lr_exponential_decrease = True
    # config.input_threads = 16
    # config.checkpoint_gap = 100

    if config.nuscene:
        config.dataset_path = 'nuScenes_like_SKitti'
    else:
        config.dataset_path = 'SemanticKitti/dataset'
    config.train_path = 'results'
    config.test_path = 'test'

    # # Get path from argument if given
    # if len(sys.argv) > 1:
    #     config.saving_path = sys.argv[1]

    # Path of the result folder
    if config.saving:
        if config.saving_path is None:
            config.saving_path = saving_path
        if not exists(config.saving_path):
            makedirs(config.saving_path, exist_ok=True)
        
        logger = logging.getLogger('main')
        logger.setLevel(logging.DEBUG)
        ### if no setFormatter or Formatter without argument, '%(message)s' is used. %(relativeCreated)d
        ### https://docs.python.org/3/library/logging.html#formatter-objects
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
        handler = logging.FileHandler(join(config.saving_path, 'output.log'))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if main_proc_glob:
            logger.info(f'config.saving path at: {config.saving_path}')
            logger.info(f'config.input_threads: {config.input_threads}')
            logger.info(f'config.first_features_dim: {config.first_features_dim}')
            logger.info(f'args.kanchor at: {args.kanchor}')

    # Initialize datasets
    training_dataset = SemanticKittiDataset(config, set='training', balance_classes=True, train_val=config.train_val)
    test_dataset = SemanticKittiDataset(config, set='validation', balance_classes=False)

    # Initialize samplers
    if distributed:
        training_sampler = SemanticKittiSamplerDistributed(training_dataset, world_size, rank)
        test_sampler = SemanticKittiSamplerDistributed(test_dataset, world_size, rank)
    else:
        training_sampler = SemanticKittiSampler(training_dataset)
        test_sampler = SemanticKittiSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=SemanticKittiCollate,
                                 num_workers=config.input_threads,
                                 #num_workers=0,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=config.input_threads,
                             #num_workers=0,
                             pin_memory=True)

    # Calibrate max_in_point value
    training_sampler.calib_max_in(config, training_loader, verbose=True)
    test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    if main_proc_glob:
        logger.info('\nModel Preparation')
        logger.info('*****************')

    # Define network model
    t1 = time.time()
    #net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)
    #print("KPFCNN")
    #net = PrototypeNet(config, training_dataset.label_values, training_dataset.ignored_labels)
    #print("Prototype")
    net = MPAnet(config, training_dataset.label_values, training_dataset.ignored_labels, world_size=world_size, rank=rank, nuscene=config.nuscene)
    if main_proc_glob:
        logger.info("MPANet")
    #net = MPAnetBinary(config, training_dataset.label_values, training_dataset.ignored_labels)
    #print("MPANetBinary")

    # Define a trainer class
    if previous_training_path:
        trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, gpu_id=gpu_id, rank=rank, distributed=distributed)
    else:
        trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, gpu_id=gpu_id, rank=rank, distributed=distributed)
    
    if distributed:
        net = DDP(net, device_ids=[gpu_id], find_unused_parameters=True)

    if main_proc_glob:
        logger.info('Done in {:.1f}s\n'.format(time.time() - t1))

        logger.info('\nStart training')
        logger.info('**************')

    # Training
    #trainer.train(net, training_loader, test_loader, config)
    #trainer.train_prototype(net, training_loader, test_loader, config)
    trainer.train_mpa(net, training_loader, test_loader, config)

    logger.info('Rank {:d}, Forcing exit now'.format(rank))
    if distributed:
        destroy_process_group()
    else:
        os.kill(os.getpid(), signal.SIGINT)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eq', action='store_true', 
                        help='If set, use equivariant mode')
    parser.add_argument('-r', '--recover', action='store_true', 
                        help='If set, recover training instead of training the second stage. ')
    parser.add_argument('-d', '--fdim', type=int, default=128, 
                        help='Set first_features_dim. ')
    parser.add_argument('-a', '--kanchor', type=int, default=1, 
                        help='Set kanchor. ')
    parser.add_argument('-n', '--nframe', type=int, default=4, 
                        help='Set n_frames. ')
             
    parser.add_argument('--head_early', action='store_true', 
                        help='If set, use inv_epn as the last layer after decoder')
    parser.add_argument('--obj_early', action='store_true', dest='early_fuse_obj',
                        help='If set, use inv_epn in the voting module')
    parser.add_argument('--share_fuse', action='store_true', dest='share_fuse_obj',
                        help='If set, use one inv_epn for both')
    parser.add_argument('--head_pool', type=str, default='mean', 
                        dest='rot_head_pool')        # max, mean, attn_best, attn_soft
    parser.add_argument('--obj_pool', type=str, default='attn_best', 
                        dest='rot_obj_pool')        # max, mean, attn_best, attn_soft
    parser.add_argument('--head_permute', action='store_true', 
                        dest='att_permute')   
    parser.add_argument('--obj_permute', action='store_true', 
                        dest='att_obj_permute')   
    parser.add_argument('--rot_sem', type=int, default=1, 
                        dest='rot_semantic_cls')
    parser.add_argument('--offset_sem', type=int, default=1, 
                        dest='offset_semantic_cls')
    parser.add_argument('--train_val', action='store_true')
    parser.add_argument('-t', '--input_threads', type=int, default=16 )
    parser.add_argument('--nuscene', action='store_true')  
    # rot_head_pool = 'mean'
    # att_permute
    # early_fuse_obj = False
    # rot_obj_pool = 'attn_best'
    # att_obj_permute = False
    # share_fuse_obj = False
    # rot_semantic_cls = 1
    # offset_semantic_cls = 1

    parser.add_argument('-l', '--log', type=str, default=None, 
                        help='In the first (pretrain) stage, do not set this argument. \
                                In the second stage, set to the training folder in the name of `Log_date_time`. ')
    parser.add_argument('-c', '--chkp', type=int, default=None, 
                        help='Set the checkpoint wanted. e.g., 200 means chkp_0200.tar. None means current_chkp.tar.')
    args = parser.parse_args()

    saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.localtime())
    gpus = torch.cuda.device_count()
    print('number of gpus', gpus)
    nodes = 1
    node_id = 0

    if gpus * nodes == 1:
        main(0, gpus, nodes, node_id, saving_path, args)
    else:
        mp.spawn(main, args=(gpus, nodes, node_id, saving_path, args), nprocs=gpus)