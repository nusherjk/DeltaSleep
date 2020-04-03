import itertools
from torch.autograd import Variable
import  numpy as np
import torch

def bbox_overlap_iou(bboxes1, bboxes2, is_anchor):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----        
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    #     import pdb; pdb.set_trace()
    x1, y1, w1, h1 = bboxes1.chunk(4, dim=-1)
    x2, y2, w2, h2 = bboxes2.chunk(4, dim=-1)

    x11 = x1 - 0.5 * w1
    y11 = y1 - 0.5 * h1
    x12 = x1 + 0.5 * w1
    y12 = y1 + 0.5 * h1
    x21 = x2 - 0.5 * w2
    y21 = y2 - 0.5 * h2
    x22 = x2 + 0.5 * w2
    y22 = y2 + 0.5 * h2

    #     x11 = torch.clamp(x11, min=0, max=1)
    #     y11 = torch.clamp(y11, min=0, max=1)
    #     x12 = torch.clamp(x12, min=0, max=1)
    #     y12 = torch.clamp(y12, min=0, max=1)
    #     x21 = torch.clamp(x21, min=0, max=1)
    #     y21 = torch.clamp(y21, min=0, max=1)
    #     x22 = torch.clamp(x22, min=0, max=1)
    #     y22 = torch.clamp(y22, min=0, max=1)


    xI1 = torch.max(x11, x21.transpose(1, 0))
    yI1 = torch.max(y11, y21.transpose(1, 0))

    xI2 = torch.min(x12, x22.transpose(1, 0))
    yI2 = torch.min(y12, y22.transpose(1, 0))

    inner_box_w = torch.clamp((xI2 - xI1), min=0)
    inner_box_h = torch.clamp((yI2 - yI1), min=0)

    inter_area = inner_box_w * inner_box_h
    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + bboxes2_area.transpose(1, 0)) - inter_area
    return torch.clamp(inter_area / union, min=0)



def Yolov2Loss(output, labels, n_truths, meta):
    B = meta['anchors']
    C = meta['classes']
    batch_size = meta['batch_size']
    threshold = meta['threshold']
    anchor_bias = meta['anchor_bias']
    scale_no_obj = meta['scale_no_obj']
    scale_coords = meta['scale_coords']
    scale_class = meta['scale_class']
    scale_obj = meta['scale_obj']

    H = output.size(2)
    W = output.size(3)

    wh = Variable(torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2]))).float()
    anchor_bias_var = Variable(torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2]))).float()

    w_list = np.array(list(range(W)), np.float32)
    wh_ids = Variable(torch.from_numpy(
        np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1,
                                                                                                        2))).float()

    zero_pad = Variable(torch.zeros(2).contiguous().view(1,2)).float()
    pad_var = Variable(torch.zeros(2 * B).contiguous().view(B, 2)).float()

    loss = Variable(torch.Tensor([0])).float()
    class_zeros = Variable(torch.zeros(C)).float()
    mask_loss = Variable(torch.zeros(H * W * B * 5).contiguous().view(H, W, B, 5)).float()
    zero_coords_loss = Variable(torch.zeros(H * W * B * 4).contiguous().view(H, W, B, 4)).float()
    zero_coords_obj_loss = Variable(torch.zeros(H * W * B * 5).contiguous().view(H, W, B, 5)).float()

    if torch.cuda.is_available():
        wh = wh.cuda()
        wh_ids = wh_ids.cuda()
        pad_var = pad_var.cuda()
        zero_pad = zero_pad.cuda()
        anchor_bias_var = anchor_bias_var.cuda()

        loss = loss.cuda()
        mask_loss = mask_loss.cuda()
        class_zeros = class_zeros.cuda()
        zero_coords_loss = zero_coords_loss.cuda()
        zero_coords_obj_loss = zero_coords_obj_loss.cuda()

    anchor_bias_var = anchor_bias_var / wh
    anchor_padded = torch.cat([pad_var, anchor_bias_var.contiguous().view(B, 2)], 1)

    predicted = output.permute(0, 2, 3, 1)
    predicted = predicted.contiguous().view(-1, H, W, B, (4 + 1 + C))

    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=4)

    adjusted_xy = sigmoid(predicted[:, :, :, :, :2])
    adjusted_obj = sigmoid(predicted[:, :, :, :, 4:5])
    adjusted_classes = softmax(predicted[:, :, :, :, 5:])

    adjusted_coords = (adjusted_xy + wh_ids) / wh

    adjusted_wh = torch.exp(predicted[:, :, :, :, 2:4]) * anchor_bias_var

    for batch in range(batch_size):

        n_true = n_truths[batch]
        if n_true == 0:
            continue

        pred_outputs = torch.cat([adjusted_coords[batch], adjusted_wh[batch]], 3)
        true_labels = labels[batch, :n_true, 1:]
        #print(pred_outputs.shape)
        #print(true_labels.shape)
        bboxes_iou = bbox_overlap_iou(pred_outputs, true_labels, False)

        # objectness loss (if iou < threshold)
        boxes_max_iou = torch.max(bboxes_iou, -1)[0]
        all_obj_mask = boxes_max_iou.le(threshold)
        all_obj_loss = all_obj_mask.unsqueeze(-1).float() * (scale_no_obj * (-1 * adjusted_obj[batch]))

        # each anchor box will learn its bias (if batch < 12800)
        all_coords_loss = zero_coords_loss.clone()
        if meta['iteration'] < 12800:
            all_coords_loss = scale_coords * torch.cat(
                [(0.5 - adjusted_xy[batch]), (0 - predicted[batch, :, :, :, 2:4])], -1)

        coord_obj_loss = torch.cat([all_coords_loss, all_obj_loss], -1)

        batch_mask = mask_loss.clone()
        truth_coord_obj_loss = zero_coords_obj_loss.clone()
        # for every true label and anchor bias
        for truth_iter in torch.arange(n_true):
            truth_iter = int(truth_iter)
            truth_box = labels[batch, truth_iter]

            #zero_pad = torch.zeros(2)
            #print(zero_pad.shape)
            #print(torch.cat([zero_pad, truth_box[3:]], 0))
            zp = torch.zeros(2)
            try:
                anchor_select = bbox_overlap_iou(torch.cat([zero_pad.t(), truth_box[3:]], 0).t(), anchor_padded, True)
            except Exception as e:
                continue
                #print(zero_pad.t().shape)
                #print(truth_box[3:].shape)
                #print(anchor_padded)

            # find the responsible anchor box
            anchor_id = torch.max(anchor_select, 1)[1]

            truth_i = (truth_box[1] * W)
            w_i = truth_i.int()
            truth_x = truth_i - w_i.float()
            truth_j = (truth_box[2] * H)
            h_j = truth_j.int()
            truth_y = truth_j - h_j.float()
            truth_wh = (truth_box[3:] / anchor_bias_var.contiguous().view(B, 2).index_select(0, anchor_id.long())).log()
            if (truth_wh[0] == Variable(- torch.FloatTensor([float('inf')]))).data[0] == 1:
                import pdb;
                pdb.set_trace()
            #padder = torch.zeros(1)
            #xpaded = torch.cat([truth_x.unsqueeze(0), padder], 0).view(1,2)
            #ypaded = torch.cat([truth_y.unsqueeze(0), padder], 0).view(1,2)
            #print(truth_y.unsqueeze(0))
            #print(truth_wh)
            truth_coords = torch.cat((truth_x.unsqueeze(0), truth_y.unsqueeze(0), truth_wh), 1)

            predicted_output = \
            predicted[batch].index_select(0, h_j.long()).index_select(1, w_i.long()).index_select(2, anchor_id.long())[
                0][0][0]
            # coords loss
            pred_xy = adjusted_xy[batch].index_select(0, h_j.long()).index_select(1, w_i.long()).index_select(2,
                                                                                                              anchor_id.long())[
                0][0][0]
            pred_wh = predicted_output[2:4]
            pred_coords = torch.cat([pred_xy, pred_wh], 0)
            #print(scale_coords.shape)
            print(truth_coords)
            print(pred_coords.unsqueeze(0))
            coords_loss = scale_coords * (truth_coords - pred_coords.unsqueeze(0))

            # objectness loss

            # given the responsible box - find iou
            iou = \
            bboxes_iou.index_select(0, h_j.long()).index_select(1, w_i.long()).index_select(2, anchor_id.long())[0][0][
                0][truth_iter]
            obj_loss = scale_obj * (iou - sigmoid(predicted_output[4]))
            truth_co_obj = torch.cat([coords_loss, obj_loss.view(1, 1)], 1)

            # class prob loss
            class_vec = class_zeros.index_fill(0, truth_box[0].long(), 1)
            class_loss = scale_class * (class_vec - torch.nn.Softmax(dim=0)(predicted_output[5:]))

            mask_ones = Variable(torch.ones(5)).float()
            if torch.cuda.is_available():
                mask_ones = mask_ones.cuda()

            batch_mask[h_j.long(), w_i.long(), anchor_id.long()] = mask_ones
            truth_coord_obj_loss[h_j.long(), w_i.long(), anchor_id.long()] = truth_co_obj

            loss += class_loss.pow(2).sum()
        # import pdb; pdb.set_trace()
        batch_coord_obj_loss = batch_mask * truth_coord_obj_loss + (1 - batch_mask) * coord_obj_loss

        loss += batch_coord_obj_loss.pow(2).sum()

    return loss