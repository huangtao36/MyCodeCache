# -*- coding:utf-8 -*-
import numpy as np


# interpolate all missing (=invalid) depths
def interpolateBackground(predicted):

    height = predicted.shape[0]
    width = predicted.shape[1]
    interpolated = predicted.copy()
    # for each row do
    for v in range(height):
        # init counter
        count = 0
        # for each pixel do
        for u in range(width):
            # if depth valid
            if predicted[v, u] > 0:
                # at least one pixel requires interpolation
                if count >= 1:
                    # first and last value for interpolation
                    u1 = u - count
                    u2 = u - 1
                    # set pixel to min depth
                    if u1 > 0 and u2 < width - 1:
                        d_ipol = predicted[v, u1 - 1] if predicted[v, u1 - 1] <= predicted[v, u2 + 1] else predicted[v, u2 + 1]
                        for u_curr in range(u1, u2 + 1, 1):
                            interpolated[v, u_curr] = d_ipol
                            # reset counter
                count = 0
            # otherwise increment counter
            else:
                count += 1
        # extrapolate to the left
        for u in range(width):
            if predicted[v, u] > 0:
                for u2 in range(u):
                    interpolated[v, u2] = predicted[v, u]
                break
        # extrapolate to the right
        for u in range(width - 1, -1, -1):
            if predicted[v, u] > 0:
                for u2 in range(u + 1, width, 1):
                    interpolated[v, u2] = predicted[v, u]
                break
    # for each column do
    for u in range(width):
        # extrapolate to the top
        for v in range(height):
            if predicted[v, u] > 0:
                for v2 in range(v):
                    interpolated[v2, u] = predicted[v, u]
                break
        # extrapolate to the bottom
        for v in range(height - 1, -1, -1):
            if predicted[v, u] > 0:
                for v2 in range(v + 1, height, 1):
                    interpolated[v2, u] = predicted[v, u]
                break
    return interpolated


def get_mask(input_matrix):
    mask = input_matrix > 0
    return mask.astype(float)


def calculate_depth_error(d_gt, d_pre):
    d_ipol = interpolateBackground(d_pre)

    gt_mask = get_mask(d_gt)
    pre_mask = get_mask(d_pre)
    mask = gt_mask * pre_mask

    pixels = np.count_nonzero(mask == 1)

    d_gt_m = d_gt * mask
    d_ipol_m = d_ipol * mask

    # --------------------------------------------------------------------- #

    d_err = np.abs(d_gt_m - d_ipol_m)
    d_err_squared = d_err * d_err

    MAE = np.sum(d_err) / pixels
    RMSE = np.sqrt(np.sum(d_err_squared) / pixels)

    # MAE = MAE * 1000.
    # RMSE = RMSE * 1000.

    # print("MAE:", MAE)
    # print("RMSE:", RMSE)

    # --------------------------------------------------------------------- #

    d_gt_m = d_gt_m.astype(float) / 1000.
    d_ipol_m = d_ipol_m.astype(float) / 1000.

    d_gt_m = np.maximum(d_gt_m, 0.0000001)
    d_ipol_m = np.maximum(d_ipol_m, 0.00000001)

    d_err_inv = np.abs((1.0 / d_gt_m - 1.0 / d_ipol_m) * mask)
    d_err_inv_squared = d_err_inv ** 2

    # print(np.max(d_err_inv))

    # gt_mask = gt_mask.flatten()
    # d_err_inv = d_err_inv.flatten()
    # d_err_inv_squared = d_err_inv_squared.flatten()

    # # d_err_inv = d_err_inv * gt_mask
    # # d_err_inv_squared = d_err_inv_squared * gt_mask

    # # print(np.count_nonzero(d_err_inv > 200))

    # imae_list = []
    # irmse_list = []
    #
    # print(np.count_nonzero(d_err_inv > 20))
    #
    # for i in range(len(d_err_inv)):
    #     if d_err_inv[i] < 20:
    #         imae_list.append(d_err_inv[i])
    #         irmse_list.append(d_err_inv_squared[i])
    #
    # # iMAE = np.average(imae_list)
    # # iRMSE = np.sqrt(np.average(irmse_list))
    #
    # count = np.count_nonzero(d_err_inv > 0)

    iMAE = np.sum(d_err_inv) / pixels
    iRMSE = np.sqrt(np.sum(d_err_inv_squared) / pixels)

    # print("iMAE:", iMAE)
    # print("iRMSE:", iRMSE)

    return iRMSE, iMAE, RMSE, MAE
