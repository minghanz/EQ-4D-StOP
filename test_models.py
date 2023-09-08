# Common libs
import signal
import os
import numpy as np
import sys
import torch
import logging
import argparse

# Dataset
from datasets.SemanticKitti import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

from models.prototype import PrototypeNet
from models.mpanet import MPAnet
from models.mpanet_binary import MPAnetBinary

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    # #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    # #
    # #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    # #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = 'results/Log_2023-02-14_05-29-48'

    # # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    # chkp_idx = 2 #None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--idx',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--log', 
        type=str,
    )
    parser.add_argument(
        '--frame', 
        type=int,
        default=4,
    )
    parser.add_argument(
        '--vis', action='store_true',
        help='save extra files to visualize the instance center votes in vis_scripts/final_vis_4d.py'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='use test set if specified, otherwise use the validation set'
    )
    parser.add_argument(
        '--name', 
        type=str,
        default=None,
    )
    args = parser.parse_args()
    
    chosen_log = args.log
    chkp_idx = args.idx

    # Choose to test on validation or test split
    on_val = not args.test #True#False

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # # Set which gpu is going to be used
    # GPU_ID = '0'
    # if torch.cuda.device_count() > 1:
    #     GPU_ID = '0, 1'

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chkp = chosen_chkp.split('.')[0]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)


    ##################################
    # Change model parameters for test
    ##################################

    config.global_fet = False
    config.validation_size = 200
    config.input_threads = 16
    config.n_frames = args.frame
    config.n_test_frames = args.frame
    # config.n_frames = 4
    #config.n_frames = 2
    # config.n_test_frames = 4
    #config.n_test_frames = 2
    if config.n_frames < config.n_test_frames:
        config.n_frames = config.n_test_frames
    config.big_gpu = True
    config.dataset_task = '4d_panoptic'
    #config.sampling = 'density'
    config.sampling = 'importance'
    config.decay_sampling = 'None'
    config.stride = 1
    config.chosen_chkp = chkp
    config.pre_train = False

    nuscene = config.nuscene

    if nuscene:
        config.dataset_path = 'nuScenes_like_SKitti'
    else:
        config.dataset_path = 'SemanticKitti/dataset'
    config.test_path = 'test'


    # Path of the result folder
    if config.saving:
        folder = args.name if args.name is not None else time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.localtime()) # time.gmtime()
        config.saving_path = join(config.test_path, config.saving_path.split('/')[-1], folder)  # so that different test runs of the same trained model are saved separately
        if not exists(config.saving_path):
            makedirs(config.saving_path)
        
        logger = logging.getLogger('main')
        logger.setLevel(logging.DEBUG)
        ### if no setFormatter or Formatter without argument, '%(message)s' is used. 
        ### https://docs.python.org/3/library/logging.html#formatter-objects
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
        handler = logging.FileHandler(join(config.saving_path, 'output.log'))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    ##############
    # Prepare Data
    ##############

    print()
    logger.info('Data Preparation')
    logger.info('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False, seqential_batch=True)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=0,#config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    logger.info('\nModel Preparation')
    logger.info('*****************')

    # Define network model
    t1 = time.time()

    #net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    #net = PrototypeNet(config, test_dataset.label_values, test_dataset.ignored_labels)
    net = MPAnet(config, test_dataset.label_values, test_dataset.ignored_labels, nuscene=nuscene)
    #net = MPAnetBinary(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    logger.info('Done in {:.1f}s\n'.format(time.time() - t1))

    logger.info('\nStart test')
    logger.info('**********\n')
    
    config.dataset_task = '4d_panoptic'
    
    # Testing

    #tester.panoptic_4d_test(net, test_loader, config)
    #tester.panoptic_4d_test_prototype(net, test_loader, config)
    tester.panoptic_4d_test_mpa(net, test_loader, config, vis=args.vis)