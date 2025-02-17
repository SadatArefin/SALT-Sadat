import argparse
import json
import time
from pathlib import Path

from models import *
from utils.datasets import *
from utils.utils import *


def test(
        cfg,
        data_cfg,
        weights,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        save_json=False,
        model=None
):
    device = torch_utils.select_device()

    # Configure run
    data_cfg_dict = parse_data_cfg(data_cfg)    # 分析配置文件，得到：训练集train.txt，测试集test.txt，names，classes等
    nC = int(data_cfg_dict['classes'])  # 一共几类
    test_path = data_cfg_dict['valid']  # 测试集位置test.txt

    if model is None:
        # Initialize model
        model = Darknet(cfg, img_size)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

    model.to(device).eval()     # 测试模式

    # Get dataloader
    # dataloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path), batch_size=batch_size)
    dataloader = LoadImagesAndLabelsAndExif(test_path, batch_size=batch_size, img_size=img_size)

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    mP, mR, mAPs, TP, jdict = [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    coco91class = coco80_to_coco91_class()

    # 修改
    for (imgs, targets, exifs, paths, shapes) in dataloader:
        targets = targets.to(device)
        t = time.time()
        output = model(imgs.to(device), exifs.to(device))
        # NMS
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)    # 4个元组对应4张图，每个都是[目标数，7]，7：4个位置,bbox_conf,类得分,哪一类

        # 计算每个样本的平均精度
        for si, detections in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            seen += 1

            if detections is None:
                # 如果有目标，但没检测到物体， AP=0
                if len(labels) != 0:
                    mP.append(0), mR.append(0), mAPs.append(0)
                continue

            # 通过置信度得分降序，对一张图所有的检测到的目标进行排序,
            detections = detections[(-detections[:, 4]).argsort()]

            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                box = detections[:, :4].clone()  # xyxy
                scale_coords(img_size, box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                # add to json dictionary
                for di, d in enumerate(detections):
                    jdict.append({
                        'image_id': int(Path(paths[si]).stem.split('_')[-1]),
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float3(d[4] * d[5])
                    })

            # 如果没有目标，但检测到了目标，AP也=0
            correct = []
            if len(labels) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mP.append(0), mR.append(0), mAPs.append(0)
                continue
            else:
                # (x, y, w, h)——> (x1, y1, x2, y2)
                target_box = xywh2xyxy(labels[:, 1:5]) * img_size
                target_cls = labels[:, 0]

                detected = []
                for *pred_box, conf, cls_conf, cls_pred in detections:
                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pred_box, target_box).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and cls_pred == target_cls[bi] and bi not in detected:
                        correct.append(1)
                        detected.append(bi)
                    else:
                        correct.append(0)

            # 计算每一类AP，# TODO P:没识别到物体P=0,R=0，AP=0,因为有物体都识别对了P=1 所以才AP=S才=R
            AP, AP_class, R, P = ap_per_class(tp=np.array(correct),
                                              conf=detections[:, 4].cpu().numpy(),
                                              pred_cls=detections[:, 6].cpu().numpy(),
                                              target_cls=target_cls.cpu().numpy())

            # 累计每一类AP，没明白
            AP_accum_count += np.bincount(AP_class, minlength=nC)   # 识别到的目标数
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)  # 识别到的目标数目标*AP

            # 计算此图像中所有类的平均AP，并附加到图像列表中
            mP.append(P.mean())
            mR.append(R.mean())
            mAPs.append(AP.mean())

            # 所有图像累计求平均P、R、AP
            mean_P = np.mean(mP)
            mean_R = np.mean(mR)
            mean_mAP = np.mean(mAPs)

        # Print image mAP and running mean mAP
        print(('%11s%11s' + '%11.3g' * 4 + 's') %
              (seen, dataloader.nF, mean_P, mean_R, mean_mAP, time.time() - t))

    # Print mAP per class
    # 这个mAP比上一个mAP大是因为，上一个是对所有AP求的平均，没检测到对象的AP=0，而这里是 对象*AP（均值），即个数*权重，没检测到对象的图片没算进来。
    # 例：AP=[1,0,1,0.5] 平均：2.5/4=0.625，只看AP,没考虑个数。但用(1*1+1*1+1*0.5)/3=0.83没考虑没检测到的图片
    print('\nmAP Per Class:')
    for i, c in enumerate(load_classes(data_cfg_dict['names'])):
        if AP_accum_count[i]:
            print('%15s: %-.4f' % (c, AP_accum[i] / (AP_accum_count[i])))
            with open('mAP.txt', 'a') as file:
                file.write(str(AP_accum[i] / (AP_accum_count[i])) + '\n')
        else:
            with open('mAP.txt', 'a') as file:
                file.write('0.00' + '\n')

    # Save JSON
    if save_json:
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO detections api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    # Return mAP
    return mean_P, mean_R, mean_mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-2cls.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/animals.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        mAP = test(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres,
            opt.save_json)
