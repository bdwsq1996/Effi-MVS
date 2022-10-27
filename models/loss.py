import torch
import numpy as np


def compute_errors(gt, pred, mask):
    # print(gt.size())
    # print(pred.size())
    mask_true_bool = gt > 0
    mask_pred_bool = pred > 0
    mask_true_bool = mask_true_bool & mask_pred_bool
    mask_true = (mask > 0.5).type(torch.float32)
    # mask_true = (gt > 425).type(torch.float32)
    # print(torch.sum(mask_true))
    mask_pred = (pred > 0).type(torch.float32)
    # print(torch.sum((pred >= -1000).type(torch.float32)))
    # print(torch.sum(mask_pred))
    mask_true = mask_true * mask_pred
    divi_gt_pred = torch.div(gt * mask_true, pred * mask_true)
    divi_pred_gt = torch.div(pred * mask_true, gt * mask_true)

    thresh = torch.max(divi_gt_pred, divi_pred_gt)

    total_num = torch.sum(mask_true)

    mask_true_thresh_0 = (thresh > 0.0).type(torch.float32)
    mask_true_thresh_1 = (thresh < 1.25).type(torch.float32)
    mask_true_thresh_2 = (thresh < 1.25 ** 2).type(torch.float32)
    mask_true_thresh_3 = (thresh < 1.25 ** 3).type(torch.float32)

    a1 = torch.sum(mask_true_thresh_0 * mask_true_thresh_1) / total_num
    a2 = torch.sum(mask_true_thresh_0 * mask_true_thresh_2) / total_num
    a3 = torch.sum(mask_true_thresh_0 * mask_true_thresh_3) / total_num

    abs_d = torch.abs(gt - pred)
    # print(abs_d.size())
    # print(abs_d[mask_true_bool].size())
    # print(total_num)
    rmse = torch.sqrt(torch.sum((abs_d ** 2) * mask_true) / total_num)

    rmse_log = torch.pow((torch.log(gt * mask_true + 1e-5) - torch.log(pred + 1e-5)), 2)
    rmse_log = torch.sum(rmse_log * mask_true) / total_num

    abs_diff = torch.sum(abs_d * mask_true) / total_num

    abs_rel = torch.sum(torch.div(abs_d, gt + 1e-7) * mask_true) / total_num

    # sq_rel = torch.sum(torch.div(torch.pow(abs_d, 2), gt + 1e-7) * mask_true) / total_num
    sq_rel = torch.std(abs_d[mask_true_bool])
    # sq_rel = abs_d.mean()
    return abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3


def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    # interval = torch.tensor(interval).view(1,1)
    # interval = interval.repeat(y_pred.size()[0], 1)
    shape = y_pred.size()
    mask_true = (y_true > 0).type(torch.float32)
    mask_pre_true = (y_pred > 0).type(torch.float32)
    mask_true = mask_true * mask_pre_true


    denom = torch.sum(mask_true) + 1e-7
    # interval_image = torch.reshape(interval, [shape[0], 1, 1, 1]).repeat(1, 1, shape[1], shape[2])
    abs_diff_image = torch.abs(y_true - y_pred) / interval
    less_three_image = mask_true * (abs_diff_image < 1.0).type(torch.float32)
    return torch.sum(less_three_image) / denom


def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    # interval = torch.tensor(interval).view(1,1)
    # interval = interval.repeat(y_pred.size()[0], 1)
    shape = y_pred.size()
    mask_true = (y_true > 0).type(torch.float32)
    mask_pre_true = (y_pred > 0).type(torch.float32)

    mask_true = mask_true * mask_pre_true
    # print(mask_true.size())
    # print(y_true.size())
    denom = torch.sum(mask_true) + 1e-7
    # interval_image = torch.reshape(interval, [shape[0], 1, 1, 1]).repeat(1, 1, shape[1], shape[2])
    abs_diff_image = torch.abs(y_true - y_pred) / interval
    # print(abs_diff_image.size())
    less_three_image = mask_true * (abs_diff_image < 3.0).type(torch.float32)
    return torch.sum(less_three_image) / denom


def mvsnet_regression_loss(depth_image, estimated_depth_image, depth_interval):
    """ compute loss and accuracy """
    # non zero mean absulote loss
    # masked_mae = non_zero_mean_absolute_diff(depth_image, estimated_depth_image, depth_interval)
    # less one accuracy
    less_one_accuracy = less_one_percentage(depth_image, estimated_depth_image, depth_interval)
    # less three accuracy
    less_three_accuracy = less_three_percentage(depth_image, estimated_depth_image, depth_interval)

    return less_one_accuracy, less_three_accuracy

