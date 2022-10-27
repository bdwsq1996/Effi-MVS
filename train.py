import argparse, os, sys, time, gc, datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import torch.distributed as dist
from datasets.data_io import read_pfm, save_pfm
import datetime
import cv2
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--mode', help='train or test')
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=48, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug/refine', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', type=bool, default=False, help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=10, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--train loss", type=str, default="0.25,0.5,1", help='last_stage_name')
parser.add_argument('--last_stage', type=str, default="stage4", help='last_stage_name')
parser.add_argument('--ndepths', type=int, default=192, help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="1", help='depth_intervals_ratio')
parser.add_argument('--GRUiters', type=str, default="3,3,3",  help='iters')
parser.add_argument('--iters', type=int, default=12,  help='iters')
parser.add_argument('--CostNum', type=int, default=1,  help='CostNum')
parser.add_argument('--trainviews', type=int, default=3,  help='trainviews')
parser.add_argument('--testviews', type=int, default=3,  help='testviews')
parser.add_argument('--logdirX', default='./checkpoints/from_old_retrain/log/', help='the directory to save checkpoints/logs')
parser.add_argument('--outdir', default='./eval_training_log', help='output dir for eval')

# main function

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, EvalImgLoader, lr_scheduler, start_epoch, args):
    logger = SummaryWriter(args.logdir)
    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx
        gru_loss = {}
        for i in range(args.iters + 1):
            gru_loss["l{}".format(i)] = 0
        # training
        print_fre = len(TrainImgLoader) // 10
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            for i in range(args.iters + 1):
                gru_loss["l{}".format(i)] += scalar_outputs["l{}".format(i)]
            # print(gru_loss)
            lr_scheduler.step()

            if batch_idx % print_fre == 0 and batch_idx > 0:
                print(
                   "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                       epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                       optimizer.param_groups[0]["lr"], loss,
                       scalar_outputs['depth_loss'],
                       time.time() - start_time))
                print(optimizer.state_dict()['param_groups'][0]['lr'])
                print(optimizer.param_groups[0]["lr"])
                print(['{}:{}'.format(key,gru_loss[key]/batch_idx) for key in gru_loss.keys()])

            del scalar_outputs


        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        # gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):

                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs_test, image_outputs = test_sample_depth(model, model_loss, sample, args)
                scalar_outputs_test['time'] = time.time() - start_time
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs_test, global_step)
                avg_test_scalars.update(scalar_outputs_test)
            del scalar_outputs_test
            # del scalar_outputs, image_outputs
            print("final", avg_test_scalars.mean())
    for i in gru_loss.keys():
        gru_loss[i] = 0

def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    i = 0
    print(len(TestImgLoader))
    for batch_idx, sample in enumerate(TestImgLoader):
        # print(batch_idx)
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        scalar_outputs['time'] = time.time() - start_time
        avg_test_scalars.update(scalar_outputs)

        del scalar_outputs, image_outputs
    print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    # num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms
    mask = mask_ms
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    outputs_depth = outputs["depth"]
    iter_list = [int(e) for e in args.GRUiters.split(",")]

    dlossw_list = [1 for x in range(iter_list[0] + 1)] + [2 for x in range(iter_list[1] +1 )] + [3 for x in range(iter_list[2] +1 )] + [4]
    loss, depth_loss_dict = model_loss(outputs_depth, depth_gt_ms, mask_ms, dlossw_list, loss_rate=0.9)


    depth_est = outputs_depth[-1]

    depth_loss = depth_loss_dict["l{}".format(args.iters)]

    loss.backward()

    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 8),}

    for i in range(args.iters + 1):
        scalar_outputs["l{}".format(i)] = depth_loss_dict["l{}".format(i)]
    image_outputs = {"depth_est": depth_est * mask[args.last_stage],
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"],
                     "errormap": (depth_est - depth_gt[args.last_stage]).abs() * mask[args.last_stage],
                     }

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    # num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms
    mask = mask_ms

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    outputs_depth = outputs["depth"]
    iter_list = [int(e) for e in args.GRUiters.split(",")]
    dlossw_list = [1 for x in range(iter_list[0] + 1)] + [2 for x in range(iter_list[1] +1 )] + [3 for x in range(iter_list[2] +1 )] + [4]
    loss, depth_loss_dict = model_loss(outputs_depth, depth_gt_ms, mask_ms, dlossw_list, loss_rate=0.9)

    depth_est = outputs_depth[-1]


    depth_loss = depth_loss_dict["l{}".format(args.iters)]

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 8),
                      "thres14mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 14),
                      "thres20mm_error": Thres_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, 20),

                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, [0, 2.0]),
                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, [2.0, 4.0]),
                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, [4.0, 8.0]),
                      "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, [8.0, 14.0]),
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, [14.0, 20.0]),
                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt[args.last_stage], mask[args.last_stage] > 0.5, [20.0, 1e5]),
                    }
    for i in range(args.iters + 1):
        scalar_outputs["l{}".format(i)] = depth_loss_dict["l{}".format(i)]

    image_outputs = {"depth_est": depth_est * mask[args.last_stage],
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"],
                     "errormap": (depth_est - depth_gt[args.last_stage]).abs() * mask[args.last_stage]}

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)

def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)
        current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        print("current time", current_time_str)
        print("creating new summary file")
        logger = SummaryWriter(args.logdir)
    print("argv:", sys.argv[1:])
    print_args(args)

    # model, optimizer
    model = Effi_MVS(args)
    model.to(device)
    model_loss = smooth_loss


    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
        weight_decay=args.wd, eps=1e-8)

    # load parameters
    start_epoch = 0
    if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        # start_epoch = 0
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    print("start at epoch {}".format(start_epoch))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)

    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.trainviews, args.numdepth)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.testviews, args.numdepth)

    TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    EvalImgLoader = None
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, len(TrainImgLoader) * args.epochs + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    # lr_scheduler = None
    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, EvalImgLoader, lr_scheduler, start_epoch, args)
    elif args.mode == "finetune":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, EvalImgLoader, lr_scheduler, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError