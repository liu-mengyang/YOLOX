#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
import numpy as np

import torch
import torch.nn.functional as F

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
    vis
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                height = imgs.shape[2]
                width = imgs.shape[3]
                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                
                r_height = 96-height%96
                r_width = 96-width%96
                # print(imgs.shape)
                imgs = F.pad(imgs, pad=(0, r_width, 0, r_height))
                # print(imgs.shape)
                height = imgs.shape[2]
                width = imgs.shape[3]
                sub_height = height / 3
                sub_width = width / 3
                imgs_sub_11 = imgs[:,:,:int(sub_height),:int(sub_width)]
                # print(imgs_sub_11.shape)
                imgs_sub_12 = imgs[:,:,:int(sub_height),int(sub_width):int(sub_width)*2]
                imgs_sub_13 = imgs[:,:,:int(sub_height),int(sub_width)*2:]
                imgs_sub_21 = imgs[:,:,int(sub_height):int(sub_height)*2,:int(sub_width)]
                imgs_sub_22 = imgs[:,:,int(sub_height):int(sub_height)*2,int(sub_width):int(sub_width)*2]
                imgs_sub_23 = imgs[:,:,int(sub_height):int(sub_height)*2,int(sub_width)*2:]
                imgs_sub_31 = imgs[:,:,int(sub_height)*2:,:int(sub_width)]
                imgs_sub_32 = imgs[:,:,int(sub_height)*2:,int(sub_width):int(sub_width)*2]
                imgs_sub_33 = imgs[:,:,int(sub_height)*2:,int(sub_width)*2:]

                outputs_sub_11 = model(imgs_sub_11)
                outputs_sub_12 = model(imgs_sub_12)
                outputs_sub_13 = model(imgs_sub_13)
                outputs_sub_21 = model(imgs_sub_21)
                outputs_sub_22 = model(imgs_sub_22)
                outputs_sub_23 = model(imgs_sub_23)
                outputs_sub_31 = model(imgs_sub_31)
                outputs_sub_32 = model(imgs_sub_32)
                outputs_sub_33 = model(imgs_sub_33)

                if decoder is not None:
                    outputs_sub_11 = decoder(outputs_sub_11, dtype=outputs_sub_11.type())
                    outputs_sub_12 = decoder(outputs_sub_12, dtype=outputs_sub_12.type())
                    outputs_sub_13 = decoder(outputs_sub_13, dtype=outputs_sub_13.type())
                    outputs_sub_21 = decoder(outputs_sub_21, dtype=outputs_sub_21.type())
                    outputs_sub_22 = decoder(outputs_sub_22, dtype=outputs_sub_22.type())
                    outputs_sub_23 = decoder(outputs_sub_23, dtype=outputs_sub_23.type())
                    outputs_sub_31 = decoder(outputs_sub_31, dtype=outputs_sub_31.type())
                    outputs_sub_32 = decoder(outputs_sub_32, dtype=outputs_sub_32.type())
                    outputs_sub_33 = decoder(outputs_sub_33, dtype=outputs_sub_33.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs_sub_11 = postprocess(
                    outputs_sub_11, self.num_classes, self.confthre, self.nmsthre
                )

                outputs_sub_12 = postprocess(
                    outputs_sub_12, self.num_classes, self.confthre, self.nmsthre
                )
                
                for i in range(len(outputs_sub_11)):
                    if outputs_sub_11[i] is not None:
                        outputs_sub_11[i] = outputs_sub_11[i].cpu()
                
                for i in range(len(outputs_sub_12)):
                    if outputs_sub_12[i] is not None:
                        outputs_sub_12[i] = outputs_sub_12[i].cpu()
                        outputs_sub_12[i][:, 0] += sub_width
                        outputs_sub_12[i][:, 2] += sub_width
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i],outputs_sub_12[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_12[i]
                outputs_sub_13 = postprocess(
                    outputs_sub_13, self.num_classes, self.confthre, self.nmsthre
                )
                for i in range(len(outputs_sub_13)):
                    if outputs_sub_13[i] is not None:
                        outputs_sub_13[i] = outputs_sub_13[i].cpu()
                        outputs_sub_13[i][:, 0] += sub_width*2
                        outputs_sub_13[i][:, 2] += sub_width*2
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i],outputs_sub_13[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_13[i]
                outputs_sub_21 = postprocess(
                    outputs_sub_21, self.num_classes, self.confthre, self.nmsthre
                )
                for i in range(len(outputs_sub_21)):
                    if outputs_sub_21[i] is not None:
                        outputs_sub_21[i] = outputs_sub_21[i].cpu()
                        outputs_sub_21[i][:, 1] += sub_height
                        outputs_sub_21[i][:, 3] += sub_height
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i],outputs_sub_21[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_21[i]
                outputs_sub_22 = postprocess(
                    outputs_sub_22, self.num_classes, self.confthre, self.nmsthre
                )
                for i in range(len(outputs_sub_22)):
                    if outputs_sub_22[i] is not None:
                        outputs_sub_22[i] = outputs_sub_22[i].cpu()
                        outputs_sub_22[i][:, 0] += sub_width
                        outputs_sub_22[i][:, 1] += sub_height
                        outputs_sub_22[i][:, 2] += sub_width
                        outputs_sub_22[i][:, 3] += sub_height
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i], outputs_sub_22[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_22[i]
                outputs_sub_23 = postprocess(
                    outputs_sub_23, self.num_classes, self.confthre, self.nmsthre
                )
                for i in range(len(outputs_sub_23)):
                    if outputs_sub_23[i] is not None:
                        outputs_sub_23[i] = outputs_sub_23[i].cpu()
                        outputs_sub_23[i][:, 0] += sub_width*2
                        outputs_sub_23[i][:, 1] += sub_height
                        outputs_sub_23[i][:, 2] += sub_width*2
                        outputs_sub_23[i][:, 3] += sub_height
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i], outputs_sub_23[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_23[i]
                outputs_sub_31 = postprocess(
                    outputs_sub_31, self.num_classes, self.confthre, self.nmsthre
                )
                for i in range(len(outputs_sub_31)):
                    if outputs_sub_31[i] is not None:
                        outputs_sub_31[i] = outputs_sub_31[i].cpu()
                        outputs_sub_31[i][:, 1] += sub_height*2
                        outputs_sub_31[i][:, 3] += sub_height*2
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i],outputs_sub_31[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_31[i]
                outputs_sub_32 = postprocess(
                    outputs_sub_32, self.num_classes, self.confthre, self.nmsthre
                )
                for i in range(len(outputs_sub_32)):
                    if outputs_sub_32[i] is not None:
                        outputs_sub_32[i] = outputs_sub_32[i].cpu()
                        outputs_sub_32[i][:, 0] += sub_width
                        outputs_sub_32[i][:, 1] += sub_height*2
                        outputs_sub_32[i][:, 2] += sub_width
                        outputs_sub_32[i][:, 3] += sub_height*2
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i], outputs_sub_32[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_32[i]
                outputs_sub_33 = postprocess(
                    outputs_sub_33, self.num_classes, self.confthre, self.nmsthre
                )
                for i in range(len(outputs_sub_33)):
                    if outputs_sub_33[i] is not None:
                        outputs_sub_33[i] = outputs_sub_33[i].cpu()
                        outputs_sub_33[i][:, 0] += sub_width*2
                        outputs_sub_33[i][:, 1] += sub_height*2
                        outputs_sub_33[i][:, 2] += sub_width*2
                        outputs_sub_33[i][:, 3] += sub_height*2
                        
                        if outputs_sub_11[i] is not None:
                            outputs_sub_11[i] = torch.cat((outputs_sub_11[i], outputs_sub_33[i]),0)
                        else:
                            outputs_sub_11[i] = outputs_sub_33[i]
                
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
            
            # test visualization
            bboxes_sub_11 = outputs_sub_11[0][:, 0:4]
            cls_sub_11 = outputs_sub_11[0][:, 6]
            scores_sub_11 = outputs_sub_11[0][:, 4] * outputs_sub_11[0][:, 5]
            cls_name = COCO_CLASSES
            vis_res_full = vis(cv2.cvtColor(np.transpose(np.array(imgs[0].cpu())/255, (1,2,0)), cv2.COLOR_RGB2BGR), bboxes_sub_11, scores_sub_11, cls_sub_11, conf=0.35, class_names=cls_name)
            plt.imshow(vis_res_full[:,:,::])
            plt.show()

            
            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs_sub_11, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            # try:
            #     from yolox.layers import COCOeval_opt as COCOeval
            # except ImportError:
            #     from pycocotools.cocoeval import COCOeval
            from pycocotools.cocoeval import COCOeval

            logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
