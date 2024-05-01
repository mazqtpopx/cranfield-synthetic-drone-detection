import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


#gets coco metrics for the pred file, based on the ground truth (gt) file
def get_coco_metrics(gt_filepath, pred_filepath):
    cocoGT = COCO(gt_filepath)
    res = cocoGT.loadRes(pred_filepath)
    annType = 'bbox'

    cocoEval = COCOeval(cocoGT,res,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval

