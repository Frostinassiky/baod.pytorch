from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

# for bi-level
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv


# for tensorboard
from tools.visualization import draw_bounding_boxes
try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

# remove old logs
os.system("rm -r .logs")
# random seed
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
torch.backends.cudnn.deterministic = True
np.random.seed(2)
def _rand_fn():
    np.random.seed(2)

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='voc_coco', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default="saved/active", type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true', default=True)
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true', default=False)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=4, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.01, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=8, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=2, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)

    # Model Path
    parser.add_argument('--model_path', dest='model_path',
                        help='Pretrained model dir',
                        default='saved/active/vgg16/voc_coco/random20/activestep3/faster_rcnn_1_10.pth', type=str)

    # active learning
    parser.add_argument('--active_step', dest='active_step',
                        help='active step',
                        default=0, type=int)
    parser.add_argument('--active_method', default='random8',
                        help='Active Method tho choose')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def bld_train(args, ann_path=None, step=0):

    # print('Train from annotaion {}'.format(ann_path))
    # print('Called with args:')
    # print(args)

    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        logger = Logger(os.path.join('./.logs', args.active_method, "/activestep" + str(step)))

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "voc_coco":
        args.imdb_name = "voc_coco_2007_train+voc_coco_2007_val"
        args.imdbval_name = "voc_coco_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    else:
        raise NotImplementedError

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)
    # np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set = source set + target set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source train set, fully labeled
    #ann_path_source = os.path.join(ann_path, 'voc_coco_2007_train_f.json')
    #ann_path_target = os.path.join(ann_path, 'voc_coco_2007_train_l.json')
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, ann_path=os.path.join(ann_path,'source'))
    imdb_tg, roidb_tg, ratio_list_tg, ratio_index_tg = combined_roidb(args.imdb_name, ann_path=os.path.join(ann_path,'target'))


    print('{:d} roidb entries for source set'.format(len(roidb)))
    print('{:d} roidb entries for target set'.format(len(roidb_tg)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + args.active_method + "/activestep" + str(step)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch_tg = None  # do not sample target set

    bs_tg = 4
    dataset_tg = roibatchLoader(roidb_tg, ratio_list_tg, ratio_index_tg, bs_tg, \
                             imdb_tg.num_classes, training=True)

    assert imdb.num_classes == imdb_tg.num_classes



    dataloader_tg = torch.utils.data.DataLoader(dataset_tg, batch_size=bs_tg,
                                             sampler=sampler_batch_tg, num_workers=args.num_workers,
                                            worker_init_fn = _rand_fn())

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    image_label = torch.FloatTensor(1)
    confidence = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        image_label = image_label.cuda()
        confidence = confidence.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    image_label = Variable(image_label)
    confidence = Variable(confidence)

    if args.cuda:
        cfg.CUDA = True

    # initialize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        raise NotImplementedError

    # initialize the expectation network.
    if args.net == 'vgg16':
        fasterRCNN_val = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN_val = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN_val = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN_val = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        raise NotImplementedError

    fasterRCNN.create_architecture()
    fasterRCNN_val.create_architecture()

    # lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise NotImplementedError

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    # expectation model
    print("load checkpoint for expectation model: %s" % args.model_path)
    checkpoint = torch.load(args.model_path)
    fasterRCNN_val.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    fasterRCNN_val = fasterRCNN_val
    fasterRCNN_val.eval()

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
        #fasterRCNN_val = nn.DataParallel(fasterRCNN_val)

    if args.cuda:
        fasterRCNN.cuda()
        fasterRCNN_val.cuda()

    # Evaluation
    # data_iter = iter(dataloader_tg)
    # for target_k in range( int(train_size_tg / args.batch_size)):
    fname = "noisy_annotations.pkl"
    if not os.path.isfile(fname):
      for batch_k, data in enumerate(dataloader_tg):
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        image_label.data.resize_(data[4].size()).copy_(data[4])
        b_size = len(im_data)
        # expactation pass
        rois, cls_prob, bbox_pred, \
        _, _, _, _, _ = fasterRCNN_val(im_data, im_info, gt_boxes, num_boxes)
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        if cfg.TRAIN.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(b_size, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    # print('DEBUG: Size of box_deltas is {}'.format(box_deltas.size()) )
                    box_deltas = box_deltas.view(b_size, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        # TODO: data distalliation
        # Choose the confident samples
        for b_idx in range(b_size):
            # fill one confidence
            # confidence.data[b_idx, :] = 1 - (gt_boxes.data[b_idx, :, 4] == 0)
            # resize prediction
            pred_boxes[b_idx] /= data[1][b_idx][2]
            for j in xrange(1, imdb.num_classes):
                if image_label.data[b_idx, j] != 1: continue  # next if no image label

                # filtering box outside of the image
                not_keep = (pred_boxes[b_idx][:, j * 4] == pred_boxes[b_idx][:, j * 4 + 2]) | \
                           (pred_boxes[b_idx][:, j * 4 + 1] == pred_boxes[b_idx][:, j * 4 + 3])
                keep = torch.nonzero(not_keep == 0).view(-1)
                # decease the number of pgts
                thresh = 0.5
                while torch.nonzero(scores[b_idx, :, j][keep] > thresh).view(-1).numel() <= 0:
                    thresh = thresh * 0.5
                inds = torch.nonzero(scores[b_idx, :, j][keep] > thresh).view(-1)

                # if there is no det, error
                if inds.numel() <= 0:
                    print('Warning!!!!!!! It should not appear!!')
                    continue

                # find missing ID
                missing_list = np.where(gt_boxes.data[b_idx, :, 4] == 0)[0]
                if (len(missing_list) == 0): continue
                missing_id = missing_list[0]
                cls_scores = scores[b_idx, :, j][keep][inds]
                cls_boxes = pred_boxes[b_idx][keep][inds][:, j * 4:(j + 1) * 4]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                keep = nms(cls_dets, 0.2)  # Magic number ????
                keep = keep.view(-1).tolist()
                sys.stdout.write('from {} predictions choose-> min({},4) as pseudo label  \r'.format(len(cls_scores), len(keep)))
                sys.stdout.flush()
                _, order = torch.sort(cls_scores[keep], 0, True)
                if len(keep) == 0: continue

                max_keep = 4
                for pgt_k in range(max_keep):
                    if len(order) <= pgt_k: break
                    if missing_id + pgt_k >= 20: break
                    gt_boxes.data[b_idx, missing_id + pgt_k, :4] = cls_boxes[keep][order[len(order) - 1 - pgt_k]]
                    gt_boxes.data[b_idx, missing_id + pgt_k, 4] = j  # class
                    #confidence[b_idx, missing_id + pgt_k] = cls_scores[keep][order[len(order) - 1 - pgt_k]]
                    num_boxes[b_idx] = num_boxes[b_idx] + 1
            sample = roidb_tg[dataset_tg.ratio_index[batch_k*bs_tg+b_idx]]
            pgt_boxes = np.array([gt_boxes[b_idx, x, :4].cpu().data.numpy() for x in range(int(num_boxes[b_idx]))])
            pgt_classes = np.array([gt_boxes[b_idx, x,4].cpu().data[0] for x in range(int(num_boxes[b_idx]))])
            sample["boxes"] = pgt_boxes
            sample["gt_classes"] = pgt_classes
            # DEBUG
            assert np.array_equal(sample["label"],image_label[b_idx].cpu().data.numpy()), \
                "Image labels are not equal! {} vs {}".format(sample["label"],image_label[b_idx].cpu().data.numpy())

      #with open(fname, 'w') as f:
      # pickle.dump(roidb_tg, f)
    else:
      pass
      # with open(fname) as f:  # Python 3: open(..., 'rb')
      # roidb_tg = pickle.load(f)

    print("-- Optimization Stage --")
    # Optimization
    print("######################################################l")


    roidb.extend(roidb_tg) # merge two datasets
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
        if True:
            if len(roidb[i]['boxes']) == 0:
                del roidb[i]
                i -= 1
        else:
            if len(roidb[i]['boxes']) == 0:
                del roidb[i]
                i -= 1
        i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    from roi_data_layer.roidb import rank_roidb_ratio
    ratio_list, ratio_index = rank_roidb_ratio(roidb)
    train_size = len(roidb)
    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers, worker_init_fn = _rand_fn())
    iters_per_epoch = int(train_size / args.batch_size)
    print("Training set size is {}".format(train_size))
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        fasterRCNN.train()

        loss_temp = 0
        start = time.time()
        epoch_start = start

        # adjust learning rate
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        # one step
        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            image_label.data.resize_(data[4].size()).copy_(data[4])

            #gt_boxes.data = \
            #    torch.cat((gt_boxes.data, torch.zeros(gt_boxes.size(0), gt_boxes.size(1), 1).cuda()), dim=2)
            conf_data = torch.zeros(gt_boxes.size(0), gt_boxes.size(1)).cuda()
            confidence.data.resize_(conf_data.size()).copy_(conf_data)

            fasterRCNN.zero_grad()

            # rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, confidence)
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            # rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, confidence)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step)

                    images = []
                    for k in range(args.batch_size):
                        image = draw_bounding_boxes(im_data[k].data.cpu().numpy(),
                                                    gt_boxes[k].data.cpu().numpy(),
                                                    im_info[k].data.cpu().numpy(),
                                                    num_boxes[k].data.cpu().numpy())
                        images.append(image)
                    logger.image_summary("Train epoch %2d, iter %4d/%4d" % (epoch, step, iters_per_epoch), \
                                          images, step)
                loss_temp = 0
                start = time.time()
                if False:
                    break

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        print('save model: {}'.format(save_name))

        epoch_end = time.time()
        print('Epoch time cost: {}'.format(epoch_end - epoch_start))

    print('finished!')


if __name__ == '__main__':
    args = parse_args()
    i = args.active_step
    div_path = os.path.join(args.active_method, 'divided')
    path = os.path.join('annotations', div_path + str(i))
    bld_train(args, path, i)
