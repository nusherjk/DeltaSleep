import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools

from Yololoss import bbox_overlap_iou, Yolov2Loss


def get_nms_boxes(output, obj_thresh, iou_thresh, meta):
    #     import pdb; pdb.set_trace()
    N, C, H, W = output.size()
    N, C, H, W = int(N), int(C), int(H), int(W)
    B = meta['anchors']
    anchor_bias = meta['anchor_bias']
    n_classes = meta['classes']

    # -1 => unprocesse, 0 => suppressed, 1 => retained
    box_tags = Variable(-1 * torch.ones(H * W * B)).float()

    wh = Variable(torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2]))).float()
    anchor_bias_var = Variable(torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2]))).float()

    w_list = np.array(list(range(W)), np.float32)
    wh_ids = Variable(torch.from_numpy(
        np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1,
                                                                                                        2))).float()

    if torch.cuda.is_available():
        wh = wh.cuda()
        wh_ids = wh_ids.cuda()
        box_tags = box_tags.cuda()
        anchor_bias_var = anchor_bias_var.cuda()

    anchor_bias_var = anchor_bias_var / wh

    predicted = output.permute(0, 2, 3, 1)
    predicted = predicted.contiguous().view(N, H, W, B, -1)
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=4)

    adjusted_xy = sigmoid(predicted[:, :, :, :, :2])
    adjusted_obj = sigmoid(predicted[:, :, :, :, 4:5])
    adjusted_classes = softmax(predicted[:, :, :, :, 5:])

    adjusted_coords = (adjusted_xy + wh_ids) / wh
    adjusted_wh = torch.exp(predicted[:, :, :, :, 2:4]) * anchor_bias_var

    batch_boxes = defaultdict()

    for n in range(N):

        scores = (adjusted_obj[n] * adjusted_classes[n]).contiguous().view(H * W * B, -1)

        class_probs = adjusted_classes[n].contiguous().view(H * W * B, -1)
        class_ids = torch.max(class_probs, 1)[1]

        pred_outputs = torch.cat([adjusted_coords[n], adjusted_wh[n]], 3)
        pred_bboxes = pred_outputs.contiguous().view(H * W * B, 4)
        ious = bbox_overlap_iou(pred_bboxes, pred_bboxes, True)

        confidences = adjusted_obj[n].contiguous().view(H * W * B)
        # get all boxes with tag -1
        final_boxes = Variable(torch.FloatTensor())
        if torch.cuda.is_available():
            final_boxes = final_boxes.cuda()

        for class_id in range(n_classes):
            bboxes_state = (
            (class_ids == class_id).float() * (scores[:, class_id] > obj_thresh).float() * box_tags).long().float()

            while (torch.sum(bboxes_state == -1) > 0).data:
                max_conf, index = torch.max(scores[:, class_id] * (bboxes_state == -1).float(), 0)
                bboxes_state = ((ious[index] < iou_thresh)[0].float() * bboxes_state).long().float()
                bboxes_state[index] = 1

                print()
                print(pred_bboxes[index].view(1,4))
                print(confidences[index].view(1, 1))
                print( class_probs[index].view(1,5))
                print()

                index_vals = torch.cat([pred_bboxes[index].view(1,4), confidences[index].view(1, 1), class_probs[index].view(1,5)],1)
                print()
                print(index_vals)
                print()
                if len(final_boxes.size()) == 0:
                    final_boxes = index_vals.view(10)
                else:
                    final_boxes = torch.cat([final_boxes, index_vals], 0)

        batch_boxes[n] = final_boxes

    return batch_boxes


# Non Max Suppression
def get_nms_detections(output, obj_thresh, iou_thresh, meta):
    #     import pdb; pdb.set_trace()
    N, C, H, W = output.size()
    N, C, H, W = int(N), int(C), int(H), int(W)
    B = meta['anchors']
    anchor_bias = meta['anchor_bias']

    # -1 => unprocesse, 0 => suppressed, 1 => retained
    box_tags = Variable(-1 * torch.ones(H * W * B)).float()

    wh = Variable(torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2]))).float()
    anchor_bias_var = Variable(torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2]))).float()

    w_list = np.array(list(range(W)), np.float32)
    wh_ids = Variable(torch.from_numpy(
        np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1,
                                                                                                        2))).float()

    if torch.cuda.is_available():
        wh = wh.cuda()
        wh_ids = wh_ids.cuda()
        box_tags = box_tags.cuda()
        anchor_bias_var = anchor_bias_var.cuda()

    anchor_bias_var = anchor_bias_var / wh

    predicted = output.permute(0, 2, 3, 1)
    predicted = predicted.contiguous().view(N, H, W, B, -1)

    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=4)

    adjusted_xy = sigmoid(predicted[:, :, :, :, :2])
    adjusted_obj = sigmoid(predicted[:, :, :, :, 4:5])
    adjusted_classes = softmax(predicted[:, :, :, :, 5:])

    adjusted_coords = (adjusted_xy + wh_ids) / wh
    adjusted_wh = torch.exp(predicted[:, :, :, :, 2:4]) * anchor_bias_var

    batch_boxes = defaultdict()

    for n in range(N):

        class_probs = adjusted_classes[n].contiguous().view(H * W * B, -1)
        pred_outputs = torch.cat([adjusted_coords[n], adjusted_wh[n]], 3)
        pred_bboxes = pred_outputs.contiguous().view(H * W * B, 4)
        ious = bbox_overlap_iou(pred_bboxes, pred_bboxes, True)

        confidences = adjusted_obj[n].contiguous().view(H * W * B)
        bboxes_state = ((confidences > obj_thresh).float() * box_tags).long().float()

        # get all boxes with tag -1
        final_boxes = Variable(torch.FloatTensor())
        if torch.cuda.is_available():
            final_boxes = final_boxes.cuda()
        while (torch.sum(bboxes_state == -1) > 0).data[0]:
            max_conf, index = torch.max(confidences * (bboxes_state == -1).float(), 0)
            bboxes_state = ((ious[index] < iou_thresh)[0].float() * bboxes_state).long().float()
            bboxes_state[index] = 1

            index_vals = torch.cat([pred_bboxes[index], confidences[index].view(1, 1), class_probs[index]], 1)
            if len(final_boxes.size()) == 0:
                final_boxes = index_vals
            else:
                final_boxes = torch.cat([final_boxes, index_vals], 0)

        batch_boxes[n] = final_boxes

    return batch_boxes


def iou_calculation(bbox1, bbox2):

	# calculates iou  between two boxes having 4 co ordinates [xbr,ybr, xtl, ytl]

	xx1 = np.maximum(bbox1[0], bbox2[0])
	yy1 = np.maximum(bbox1[1], bbox2[1])
	xx2 = np.minimum(bbox1[2], bbox2[2])
	yy2 = np.maximum(bbox1[3], bbox2[3])

	w = np.maximum(0, xx2-xx1)
	h = np.maximum(0, yy2-yy1)

	area = w*h

	iou = area/ (((bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])) + ((bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])) - area)

	return iou


def calc_map(boxes_dict, iou_threshold=0.5):
    #     import pdb; pdb.set_trace()
    v = Variable(torch.zeros(1))
    if torch.cuda.is_available():
        v = v.cuda()

    if (len(boxes_dict['ground_truth']) == 0) or (len(boxes_dict['prediction']) == 0):
        return v

    gt = boxes_dict['ground_truth']
    pr = boxes_dict['prediction']
    print()
    print(len(pr))
    print(len(gt))
    print()

    gt_matched = Variable(-torch.ones(gt.size(0)))
    pr_matched = Variable(-torch.ones(pr.size(0)))

    if torch.cuda.is_available():
        gt_matched = gt_matched.cuda()
        pr_matched = pr_matched.cuda()

    for i in range(len(pr)):
        b = pr[i]
        print()
        print("b:")
        print(b)
        print("####")
        print()

        ious = bbox_overlap_iou(b[:4].view(1, 4), gt, True)

        matched_scores = (gt_matched == -1).float() * (ious[0] > iou_threshold).float() * ious[0]
        if torch.sum(matched_scores).data[0] > 0:
            gt_idx = torch.max(matched_scores, 0)[1]
            gt_matched[gt_idx] = i
            pr_matched[i] = gt_idx

    tp = (pr_matched != -1).float()
    fp = (pr_matched == -1).float()
    tp_cumsum = torch.cumsum(tp, 0)
    fp_cumsum = torch.cumsum(fp, 0)
    n_corrects = tp_cumsum * tp
    total = tp_cumsum + fp_cumsum
    precision = n_corrects / total
    for i in range(precision.size(0)):
        precision[i] = torch.max(precision[i:])

    average_precision = torch.sum(precision) / len(gt)
    return average_precision


def evaluation(ground_truths, nms_output, n_truths, iou_thresh):
    #     import pdb; pdb.set_trace()
    N = ground_truths.size(0)

    mean_avg_precision = Variable(torch.FloatTensor([0]))
    if torch.cuda.is_available():
        mean_avg_precision = mean_avg_precision.cuda()

    for batch in range(int(N)):
        category_map = defaultdict(lambda: defaultdict(lambda: torch.FloatTensor()))

        if n_truths[batch] == 0:
            continue

        ground_truth = ground_truths[batch, :n_truths[batch]]
        for gt in ground_truth:
            gt_class = gt[0].int().data
            t1 = category_map[gt_class]['ground_truth']
            if len(t1.size()) == 0:
                t1 = gt[1:].unsqueeze(0)
            else:
                t1 = torch.cat([t1, gt[1:].unsqueeze(0)], 0)
            category_map[gt_class]['ground_truth'] = t1

        nms_boxes = nms_output[batch]
        if len(nms_boxes.size()) == 0:
            continue

        for box in nms_boxes:
            class_id = (torch.max(box[5:], 0)[1]).int().data
            t2 = category_map[class_id]['prediction']
            if len(t2.size()) == 0:
                t2 = box[:5].unsqueeze(0)
            else:
                t2 = torch.cat([t2, box[:5].unsqueeze(0)], 0)
            category_map[class_id]['prediction'] = t2
        cat_ids = category_map.keys()
        #         return category_map
        dd = list()
        for cat_id in cat_ids:
            catmap = category_map[cat_id]
            map = calc_map(catmap, iou_thresh)
            dd.append(map)
        dd = np.asarray(dd)

        tt = torch.from_numpy(dd)
        mean_avg_precision += torch.mean(tt)
        #print("this worked like a charm")
        #mean_avg_precision += torch.mean(
        #    torch.cat([calc_map(category_map[cat_id], iou_thresh) for cat_id in cat_ids], 0))
    return mean_avg_precision / N

def get_map(output, labels, n_true, iou_thresh, meta):
    nms_output = get_nms_boxes(output, 0.24, 0.3, meta)
    mean_avg_prec = evaluation(labels, nms_output, n_true.numpy(), iou_thresh)
    return mean_avg_prec

'''
def train(model, train_data, opt, iou_thresh):
    train_images = Variable(train_data["image"], requires_grad=True).cuda().float()
    train_labels = Variable(train_data["bboxes"], requires_grad=False).cuda().float()
    train_n_true = train_data["n_true"]
    opt.zero_grad()
    train_output = model(train_images)
    loss = Yolov2Loss(train_output, train_labels, train_n_true.numpy())
    loss.backward()
    opt.step()
    train_map = get_map(train_output, train_labels, train_n_true, iou_thresh)
    return loss, train_map
'''

def validate(model, test_data, iou_thresh):
    test_images = Variable(test_data["image"]).cuda().float()
    test_labels = Variable(test_data["bboxes"]).cuda().float()
    test_n_true = test_data["n_true"]

    test_output = model(test_images)
    test_loss = Yolov2Loss(test_output, test_labels, test_n_true.numpy())
    test_map = get_map(test_output, test_labels, test_n_true, iou_thresh)
    return test_loss, test_map



def draw_boundary_box(image_dict):
    """Show image with landmarks"""
    image = image_dict['image']
    bboxes = image_dict['bboxes']
    plt.imshow(image)
    for bbox in bboxes:
        xmin = bbox[1]
        ymin = bbox[2]
        xmax = bbox[3]
        ymax = bbox[4]
        plt.plot((xmin, xmin), (ymin, ymax), 'g')
        plt.plot((xmin, xmax), (ymin, ymin), 'g')
        plt.plot((xmax, xmax), (ymin, ymax), 'g')
        plt.plot((xmin, xmax), (ymax, ymax), 'g')

def draw_boundary_box_fraction(image_dict):
    """Show image with landmarks"""
    image = image_dict['image']
    height, width, channels = image.shape
    bboxes = image_dict['bboxes']
    plt.imshow(image)
    for bbox in bboxes:
        x = bbox[1] * width
        y = bbox[2] * height
        w = bbox[3] * width
        h = bbox[4] * height
        xmin = max(0, int(x - w*0.5))
        ymin = max(0, int(y - h*0.5))
        xmax = min(int(x + w*0.5), width)
        ymax = min(int(y + h*0.5), height)
        plt.plot((xmin, xmin), (ymin, ymax), 'g')
        plt.plot((xmin, xmax), (ymin, ymin), 'g')
        plt.plot((xmax, xmax), (ymin, ymax), 'g')
        plt.plot((xmin, xmax), (ymax, ymax), 'g')

def draw_bbox_torch(img_dict):
    image = np.transpose(np.array(img_dict['image'].numpy()*255, np.uint8), (1, 2, 0))
    bboxes = img_dict['bboxes'].numpy()
    n_true = np.sum(bboxes[:, 0]!=-1)
    bboxes = bboxes[:n_true]
    draw_boundary_box_fraction({"image": image, 'bboxes': bboxes})

def draw_bbox_nms(torch_img, nms_output):
    image = np.transpose(np.array(torch_img.data.numpy()*255, np.uint8), (1, 2, 0))
    bboxes = nms_output[:, :4].data.numpy()
    bboxes = np.concatenate([np.zeros(len(bboxes)).reshape(len(bboxes), 1), bboxes], 1)
    print (bboxes)
    draw_boundary_box_fraction({"image": image, 'bboxes': bboxes})