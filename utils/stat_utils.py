from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from utils.regions import *

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def analyze(ref_introns, pred_introns):
    #true positive introns
    ref_intersect_pred = set(ref_introns).intersection(pred_introns)
    #false negative introns
    ref_differ_pred = set(ref_introns).difference(pred_introns)
    #false positive introns
    pred_differ_ref = set(pred_introns).difference(ref_introns)
    true_positive = float(len(ref_intersect_pred))
    false_positive = float(len(pred_differ_ref))
    false_negative = float(len(ref_differ_pred))
    sensitivity = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    logging.debug ("%.0f %.0f %.0f %.4f %.4f" % (true_positive, false_positive, false_negative, sensitivity, precision))
    return sensitivity, precision, ref_intersect_pred, ref_differ_pred, pred_differ_ref
