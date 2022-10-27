import argparse, os, time, sys, gc, cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from functools import partial
import signal

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=96, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs_cas', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=int, default=48, help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--maskupsample', type=str, default="last",  help='maskupsample')
parser.add_argument('--hiddenstate', type=str, default="init",  help='hiddenstate')
parser.add_argument('--GRUiters', type=str, default="3,3,3",  help='iters')
parser.add_argument('--CostNum', type=int, default=4,  help='CostNum')

parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=1184, help='testing max h')
parser.add_argument('--max_w', type=int, default=1600, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=1, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")

#filter
parser.add_argument('--conf', type=float, default=0.3, help='prob confidence')
parser.add_argument('--data_type', type=str, default='dtu', help='prob confidence')
parser.add_argument('--thres_view', type=int, default=2, help='threshold of num view')

#filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='../fusibile/fusibile')



# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

# num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
num_stage = 3
Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)
def prepare_img(hr_img):
    # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

    # downsample
    h, w = hr_img.shape
    hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    # crop
    h, w = hr_img_ds.shape
    target_h, target_w = 512, 640
    start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
    hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

    # #downsample
    # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

    return hr_img_crop
def read_mask_hr(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32)
    np_img = (np_img > 10).astype(np.float32)
    np_img = prepare_img(np_img)

    h, w = np_img.shape

    return (np_img > 0.5)
# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] *= 2
    depth_min = float(lines[11].split()[3])
    depth_max = float(lines[11].split()[2])
    if depth_max>425:
        depth_max = 935
        depth_min = 425
    return intrinsics, extrinsics, depth_max, depth_min


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam, depth_max, depth_min):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(depth_max) + ' ' + str(depth_min) + '\n')

    f.close()

def save_depth(testlist):

    for scene in testlist:
        save_scene_depth([scene])

# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    if args.data_type == 'dtu':
        test_dataset = MVSDataset(args.testpath, testlist, "test", args.num_view, args.numdepth, max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    elif args.data_type == 'tank':
        test_dataset = MVSDataset(args.testpath, args.num_view, args.numdepth, scan=testlist)
    else:
        print("wrong data_type")
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    model = Effi_MVS(args)
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=False)
    # model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            depth_max = 1. / sample["depth_values"][:, 0]
            depth_min = 1. / sample["depth_values"][:, -1]

            sample_cuda = tocuda(sample)
            depth_max = tensor2numpy(depth_max)
            depth_min = tensor2numpy(depth_min)
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            torch.cuda.synchronize()
            end_time = time.time()
            del sample_cuda
            torch.cuda.empty_cache()
            outputs = tensor2numpy(outputs)
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage4"].numpy()
            imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            # # save depth maps and confidence maps
            for filename, cam, img, depth_est, photometric_confidence, depth_max_, depth_min_ in zip(filenames, cams, imgs, outputs["depth"][-1], outputs["photometric_confidence"], depth_max, depth_min):
                img = img[0]  #ref view
                _, h, w = img.shape
                cam = cam[0]  #ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                #
                print(depth_est.shape)
                # print(depth_est.mean())
                # print(depth_est.max())
                # print(depth_est.min())
                #
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                #save depth maps
                save_pfm(depth_filename, depth_est)
                #save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                #save cams, img
                write_cam(cam_filename, cam, depth_max_, depth_min_)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # print(img_bgr.shape)
                cv2.imwrite(img_filename, img_bgr)

    torch.cuda.empty_cache()
    gc.collect()


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected = np.where(K_xyz_reprojected == 0, 1e-5, K_xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    xy_reprojected = np.clip(xy_reprojected, -1e8, 1e8)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency_tank(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,dh_pixel_dist_num):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks = []
    for i in range(dh_pixel_dist_num[0], 11):
        mask = np.logical_and(dist < i / dh_pixel_dist_num[1], relative_depth_diff < i / dh_pixel_dist_num[2])
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src


def filter_depth_tank(scan, pair_folder, scan_folder, out_folder, plyfilename):
    if scan in ['Family','Francis', 'Horse', 'Lighthouse','M60', 'Panther', 'Playground', 'Train']:
        dh_view_num = 3
        dh_pixel_dist_num = [dh_view_num,8,1600]
    else:
        dh_view_num = 3
        dh_pixel_dist_num = [dh_view_num,4,800]
    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    ct2 = -1

    for ref_view, src_views in pair_data:

        ct2 += 1

        # load the camera parameters
        ref_intrinsics, ref_extrinsics, ref_depth_max, ref_depth_min = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        h,w = ref_depth_est.shape
        confidence = cv2.resize(confidence, (int(w), int(h)))

        photo_mask = confidence > args.conf

        all_srcview_depth_ests = []

        # compute the geometric mask
        geo_mask_sum = 0
        geo_mask_sums = []
        ct = 0
        for src_view in src_views:
            ct = ct + 1
            # camera parameters of the source view
            src_intrinsics, src_extrinsics, src_depth_max, src_depth_min  = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]


            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency_tank(ref_depth_est,
                                                                                               ref_intrinsics,
                                                                                               ref_extrinsics,
                                                                                               src_depth_est,
                                                                                               src_intrinsics,
                                                                                               src_extrinsics,
                                                                                               dh_pixel_dist_num)

            if (ct == 1):
                for i in range(dh_view_num, 11):
                    geo_mask_sums.append(masks[i - dh_view_num].astype(np.int32))
            else:
                for i in range(dh_view_num, 11):
                    geo_mask_sums[i - dh_view_num] += masks[i - dh_view_num].astype(np.int32)

            geo_mask_sum += geo_mask.astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)

        geo_mask = geo_mask_sum >= 10
        geo_mask2 = geo_mask
        for i in range(dh_view_num, 11):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - dh_view_num] >= i)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        maskdepth = np.logical_and(depth_est_averaged > ref_depth_min, depth_est_averaged < ref_depth_max)
        if (not isinstance(geo_mask, bool)):

            final_mask = np.logical_and(photo_mask, geo_mask)
            final_mask = np.logical_and(final_mask, maskdepth)
            os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)

            save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

            print("processing {}, ref-view{:0>2}, photo/geo/geo2/final-mask:{}/{}/{}/{}".format(scan_folder, ref_view,
                                                                                        photo_mask.mean(),
                                                                                        geo_mask.mean(),
                                                                                        geo_mask2.mean(),
                                                                                        final_mask.mean()))

            if args.display:
                cv2.imshow('ref_img', ref_img[:, :, ::-1])
                cv2.imshow('ref_depth', ref_depth_est / 800)
                cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
                cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
                cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
                cv2.waitKey(0)

            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            valid_points = final_mask
            print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            color = ref_img[:, :, :][valid_points]
            # color = ref_img[1::2, 1::2, :][valid_points]
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]
            vertexs.append(xyz_world.transpose((1, 0)))
            vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, ref_depth_max, ref_depth_min):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)

    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)


    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask2 = np.logical_and(depth_ref > ref_depth_min, depth_ref < ref_depth_max)
    # mask = np.logical_and(dist < 0.2, relative_depth_diff < 0.0025)
    mask = np.logical_and(dist < 0.125, depth_diff < 0.125)


    depth_reprojected[~mask2] = 0
    depth_reprojected[~mask] = 0
    mask = np.logical_and(mask, mask2)


    return mask, depth_reprojected, x2d_src, y2d_src
# # #
# # #
def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:

        ref_intrinsics, ref_extrinsics, ref_depth_max, ref_depth_min = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))

        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]

        h,w = ref_depth_est.shape

        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]


        confidence = cv2.resize(confidence, (int(w), int(h)))


        photo_mask = confidence > args.conf

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        # compute the geometric mask
        geo_mask_sum = 0
        for i, src_view in enumerate(src_views):
            # camera parameters of the source view
            src_intrinsics, src_extrinsics, src_depth_max, src_depth_min = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            # src_depth_est = cv2.resize(src_depth_est, (int(w * 2), int(h * 2)))
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics, ref_depth_max, ref_depth_min)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= args.thres_view
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        if num_stage == 1:
            color = ref_img[1::4, 1::4, :][valid_points]
        elif num_stage == 2:
            color = ref_img[1::2, 1::2, :][valid_points]
        elif num_stage == 3:
            color = ref_img[valid_points]
            # color = ref_img[1::2, 1::2, :][valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)



if __name__ == '__main__':
    if args.data_type == 'dtu':
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        #for tanks & temples or eth3d or colmap
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    #
    if args.data_type == 'tank':
        testlist = ['Family', 'Francis', 'Horse', 'Lighthouse','M60', 'Panther', 'Playground', 'Train', 'Auditorium', 'Ballroom', 'Courtroom','Museum', 'Palace', 'Temple']
        save_depth(testlist)
    elif args.data_type == 'dtu':
        save_depth(testlist)
    else:
        print("wrong data_type")


    if args.data_type == 'tank':
        testlist = ['Family', 'Francis', 'Horse', 'Lighthouse','M60', 'Panther', 'Playground', 'Train', 'Auditorium', 'Ballroom', 'Courtroom','Museum', 'Palace', 'Temple']
    for scan in testlist:
        if args.data_type == 'tank':
            if scan in ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']:
                path = args.testpath + 'intermediate/'
            else:
                path = args.testpath + 'advanced/'
            pair_folder = os.path.join(path, scan)
        elif  args.data_type == 'dtu':
            path = args.testpath
            pair_folder = os.path.join(args.testpath, scan)
            scan_id = int(scan[4:])
        else:
            print("wrong data_type")
        scan_folder = os.path.join(args.outdir, scan)
        out_folder = os.path.join(args.outdir, scan)
        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        plypath = args.outdir + '/plyfilter'
        if not os.path.exists(plypath):
            os.makedirs(plypath)
        if args.data_type == 'tank':
            filter_depth_tank(scan, pair_folder, scan_folder, out_folder, os.path.join(args.outdir, 'plyfilter/{}.ply'.format(scan)))
            # filter_depth(pair_folder, scan_folder, out_folder,os.path.join('/home2/wangshaoqian/home2/wangshaoqian/wsq/MVS/cascade/CasMVSNet/tank0831_easy_costnet_333_inverse_stage_channel_maxandmin_depth96_CostNum4_view7_numdepth384_maskinverse_stage4_8421/plynewall_2to11_03_im_4_4000_ad_3_1000', '{}.ply'.format(scan)))
        elif args.data_type == 'dtu':
            filter_depth(pair_folder, scan_folder, out_folder,os.path.join(args.outdir, 'plyfilter/mvsnet{:0>3}_l3.ply'.format(scan_id)))
        else:
            print("wrong data_type")