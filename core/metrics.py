# #NEW CODE 
# import numpy as np


# class FewShotMetric(object):
#     def __init__(self, n_class):
#         # Ignore n_class; track binary IoU (defect vs. background)
#         self.n_class = n_class  # Kept for compatibility
#         self.stat = np.zeros(7)  # [tp, fp, fn , good_catch , positive_pairs , correct_yeild , negative_yeild]

#     def update(self, pred, ref, cls, ori_size=None, verbose=0):
#         """
#         Update metrics for Vision24 defect segmentation (.binary IoU).

#         Args:
#             pred: Predicted masks [batch, H, W] or [batch, 1, H, W]
#             ref: Ground truth masks [batch, H, W], expected to be binary (0, 1)
#             cls: Class indices (ignored)
#             ori_size: Original sizes [(H, W), ...] (optional)
#             verbose: If > 0, print IoU per sample
#         """
#         pred = np.asarray(pred, np.uint8)
#         ref = np.asarray(ref, np.uint8)

#         # Validate ground truth: must be binary (0, 1)
#         unique_ref = np.unique(ref)
#         if not np.all(np.isin(unique_ref, [0, 1])):
#             raise ValueError(f"Ground truth masks must be binary (0, 1), got values: {unique_ref}")

#         # Convert predictions to binary (threshold > 0)
#         pred = (pred > 0).astype(np.uint8)

#         for i in range(pred.shape[0]):  # Iterate over batch
#             p = pred[i]
#             r = ref[i]

#             if ori_size is not None:
#                 ori_H, ori_W = ori_size[i]
#                 p = p[:ori_H, :ori_W]
#                 r = r[:ori_H, :ori_W]

#             # Binary IoU calculation
#             tp = np.sum((p == 1) & (r == 1))  # True positives
#             fp = np.sum((p == 1) & (r == 0))  # False positives
#             fn = np.sum((p == 0) & (r == 1))  # False negatives

#             #iou
#             iou = tp / (tp + fp + fn + 1e-7)

#             #yeild rate
#             if(np.sum((r == 1)) == 0 ):
#                 #find the all white value in the predict
#                 defect_in_pred = np.sum((p == 1))
#                 #all mask area
#                 all_mask = np.sum((p == 1) & (p == 0))

#                 #percentage of white space to all space
#                 percentage  = (defect_in_pred) / (all_mask)
                
#                 #Adding for the negative pairs , where the mask with no defect 
#                 self.stat[6] += 1
                
#                 if(precentage <= 0.000239):
#                     self.stat[5] += 1  # good yeild
#             else:
#                 # the number of positive pairs ( where the mask has a defect)
#                 self.stat[4] +=1
                
#                 iou = tp / (tp + fp + fn + 1e-7)

                
#                 if(iou  >= 0.3):
#                     #number of good catch where the defect was detected with IOU >= 0.3
#                     self.stat[3] +=1 
            
#             if verbose:
#                 #This is a debut if statement
#                 iou = tp / (tp + fp + fn + 1e-7)         
#                 print(f"Sample {i} IoU: {iou:.4f}")

#             self.stat[0] += tp
#             self.stat[1] += fp
#             self.stat[2] += fn

#     def get_scores(self, labels, binary=False):
#         """
#         Compute binary IoU.

#         Args:
#             labels: Ignored, kept for compatibility
#             binary: Ignored, always True for Vision24

#         Returns:
#             mIoU_class: np.ndarray, [iou] (single IoU value)
#             mean: float, Same as iou (for compatibility)
#         """
#         tp, fp, fn ,good_catch,postive_pairs , correct_yeild , negative_pairs = self.stat
#         iou = tp / (tp + fp + fn + 1e-7)

#         catch_rate=  (good_catch) / (postive_pairs)
#         yeild_rate = (correct_yeild) / (negative_pairs)
        
#         return np.array([iou]), iou, catch_rate, yeild_rate

#     def reset(self):
#         self.stat = np.zeros(7)


# class SegmentationMetric(object):
#     def __init__(self, n_class):
#         # Ignore n_class; use 2x2 confusion matrix for binary IoU
#         self.n_class = n_class  # Kept for compatibility
#         self.confusion_matrix = np.zeros((2, 2))  # [background, foreground]

#     def _fast_hist(self, label_true, label_pred):
#         # Binary histogram (0, 1)
#         mask = (label_true >= 0) & (label_true <= 1)
#         hist = np.bincount(
#             2 * label_true[mask].astype(int) + label_pred[mask],
#             minlength=4,
#         ).reshape(2, 2)
#         return hist

#     def update(self, pred, ref, ori_size=None):
#         """
#         Update confusion matrix for binary IoU.

#         Args:
#             pred: Predicted masks [batch, H, W] or [batch, 1, H, W]
#             ref: Ground truth masks [batch, H, W], binary (0, 1)
#             ori_size: Original sizes [(H, W), ...] (optional)
#         """
#         pred = np.asarray(pred, np.uint8)
#         ref = np.asarray(ref, np.uint8)

#         # Validate ground truth
#         unique_ref = np.unique(ref)
#         if not np.all(np.isin(unique_ref, [0, 1])):
#             raise ValueError(f"Ground truth masks must be binary (0, 1), got values: {unique_ref}")

#         pred = (pred > 0).astype(np.uint8)

#         for i, (r, p) in enumerate(zip(ref, pred)):
#             if ori_size is not None:
#                 ori_H, ori_W = ori_size[i]
#                 p = p[:ori_H, :ori_W]
#                 r = r[:ori_H, :ori_W]
#             self.confusion_matrix += self._fast_hist(r.flatten(), p.flatten())

#     def get_scores(self, binary=False, withbg=True):
#         """
#         Compute binary IoU from confusion matrix.

#         Args:
#             binary: Ignored, always True
#             withbg: If True, return [bg_iou, fg_iou]; else, [fg_iou]

#         Returns:
#             iou: np.ndarray, IoU values
#             mean_iou: float, Mean IoU
#         """
#         hist = self.confusion_matrix
#         iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-7)
#         if not withbg:
#             iou = iou[1:]
#         mean_iou = np.nanmean(iou)
#         return iou, mean_iou

#     def reset(self):
#         self.confusion_matrix = np.zeros((2, 2))


# class Accumulator(object):
#     def __init__(self, **kwargs):
#         self.values = kwargs
#         self.counter = {k: 0 for k, v in kwargs.items()}
#         for k, v in self.values.items():
#             if not isinstance(v, (float, int, list)):
#                 raise TypeError(f"The Accumulator does not support `{type(v)}`. Supported types: [float, int, list]")

#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             if isinstance(self.values[k], list):
#                 self.values[k].append(v)
#             else:
#                 self.values[k] = self.values[k] + v
#             self.counter[k] += 1

#     def reset(self):
#         for k in self.values.keys():
#             if isinstance(self.values[k], list):
#                 self.values[k] = []
#             else:
#                 self.values[k] = 0
#             self.counter[k] = 0

#     def mean(self, key, axis=None, dic=False):
#         if isinstance(key, str):
#             if isinstance(self.values[key], list):
#                 return np.array(self.values[key]).mean(axis)
#             else:
#                 return self.values[key] / (self.counter[key] + 1e-7)
#         elif isinstance(key, (list, tuple)):
#             if dic:
#                 return {k: self.mean(k, axis) for k in key}
#             return [self.mean(k, axis) for k in key]
#         else:
#             TypeError(f"`key` must be a str/list/tuple, got {type(key)}")

#     def std(self, key, axis=None, dic=False):
#         if isinstance(key, str):
#             if isinstance(self.values[key], list):
#                 return np.array(self.values[key]).std(axis)
#             else:
#                 raise RuntimeError("`std` is not supported for (int, float). Use list instead.")
#         elif isinstance(key, (list, tuple)):
#             if dic:
#                 return {k: self.std(k) for k in key}
#             return [self.std(k) for k in key]
#         else:
#             TypeError(f"`key` must be a str/list/tuple, got {type(key)}")

#     def get(self):
#         return {k: v.copy() if isinstance(v, list) else v for k, v in self.values.items()}
    
#     def load(self, data):
#         for k, v in data.items():
#             if k in self.values:
#                 self.values[k] = v.copy() if isinstance(v, list) else v
#                 self.counter[k] = len(v) if isinstance(v, list) else 0
import numpy as np


import numpy as np

class FewShotMetric(object):
    def __init__(self, n_class):
        self.n_class = n_class  # Number of defect classes (e.g., 13 for 0 to 12)
        self.stat = np.zeros(7)  # [tp, fp, fn, good_catch, positive_pairs, correct_yield, negative_pairs]
        self.class_stat = np.zeros((self.n_class, 3))  # [tp, fp, fn] for each class

    def update(self, pred, ref, cls, ori_size=None, verbose=0):
        """
        Update metrics for Vision24 defect segmentation (binary IoU and per-class IoU).

        Args:
            pred: Predicted masks [batch, H, W], binary (0 or 1)
            ref: Ground truth masks [batch, H, W], binary (0 or 1)
            cls: List or array of class indices for each sample in the batch
            ori_size: Original sizes [(H, W), ...] (optional)
            verbose: If > 0, print binary IoU per sample
        """
        pred = np.asarray(pred, np.uint8)
        ref = np.asarray(ref, np.uint8)
        cls = np.asarray(cls)

        # Validate ground truth: must be binary (0, 1)
        unique_ref = np.unique(ref)
        if not np.all(np.isin(unique_ref, [0, 1])):
            raise ValueError(f"Ground truth masks must be binary (0, 1), got values: {unique_ref}")

        for i in range(pred.shape[0]):  # Iterate over batch
            p = pred[i]
            r = ref[i]
            c = cls[i]  # Class index for this sample

            if ori_size is not None:
                ori_H, ori_W = ori_size[i]
                p = p[:ori_H, :ori_W]
                r = r[:ori_H, :ori_W]

            # Binary IoU calculation for overall defect vs. background
            tp_binary = np.sum((p == 1) & (r == 1))
            fp_binary = np.sum((p == 1) & (r == 0))
            fn_binary = np.sum((p == 0) & (r == 1))

            self.stat[0] += tp_binary
            self.stat[1] += fp_binary
            self.stat[2] += fn_binary

            # Per-class IoU calculation using cls parameter
            if c >= 0 and c < self.n_class:
                self.class_stat[c, 0] += tp_binary
                self.class_stat[c, 1] += fp_binary
                self.class_stat[c, 2] += fn_binary

            # Yield rate and catch rate calculations
            if np.sum(r) == 0:  # No defect in ground truth
                self.stat[6] += 1  # negative_pairs
                defect_in_pred = np.sum(p)
                all_mask = p.size
                percentage = defect_in_pred / all_mask
                if percentage <= 0.000239:
                    self.stat[5] += 1  # correct_yield
            else:  # Defect present in ground truth
                self.stat[4] += 1  # positive_pairs
                iou_binary = tp_binary / (tp_binary + fp_binary + fn_binary + 1e-7)
                if iou_binary >= 0.3:
                    self.stat[3] += 1  # good_catch

            if verbose:
                iou_binary = tp_binary / (tp_binary + fp_binary + fn_binary + 1e-7)
                print(f"Sample {i} IoU: {iou_binary:.4f}")

    def get_scores(self, labels, binary=False):
        """
        Compute binary IoU and per-class IoU.

        Args:
            labels: Ignored, kept for compatibility
            binary: Ignored, kept for compatibility

        Returns:
            mIoU_class: np.ndarray, IoU for each class (0 to n_class-1)
            mean: float, Binary IoU (defect vs. background)
            catch_rate: float, Catch rate based on binary IoU
            yield_rate: float, Yield rate based on binary IoU
        """
        tp, fp, fn, good_catch, positive_pairs, correct_yield, negative_pairs = self.stat
        iou_binary = tp / (tp + fp + fn + 1e-7)
        catch_rate = good_catch / (positive_pairs + 1e-7)
        yield_rate = correct_yield / (negative_pairs + 1e-7)

        # Per-class IoU
        class_iou = self.class_stat[:, 0] / (self.class_stat[:, 0] + self.class_stat[:, 1] + self.class_stat[:, 2] + 1e-7)

        return class_iou, iou_binary, catch_rate, yield_rate

    def reset(self):
        self.stat = np.zeros(7)
        self.class_stat = np.zeros((self.n_class, 3))
  
class SegmentationMetric(object):
    def __init__(self, n_class):
        # Ignore n_class; use 2x2 confusion matrix for binary IoU
        self.n_class = n_class  # Kept for compatibility
        self.confusion_matrix = np.zeros((2, 2))  # [background, foreground]

    def _fast_hist(self, label_true, label_pred):
        # Binary histogram (0, 1)
        mask = (label_true >= 0) & (label_true <= 1)
        hist = np.bincount(
            2 * label_true[mask].astype(int) + label_pred[mask],
            minlength=4,
        ).reshape(2, 2)
        return hist

    def update(self, pred, ref, ori_size=None):
        """
        Update confusion matrix for binary IoU.

        Args:
            pred: Predicted masks [batch, H, W] or [batch, 1, H, W]
            ref: Ground truth masks [batch, H, W], binary (0, 1)
            ori_size: Original sizes [(H, W), ...] (optional)
        """
        pred = np.asarray(pred, np.uint8)
        ref = np.asarray(ref, np.uint8)

        # Validate ground truth
        unique_ref = np.unique(ref)
        if not np.all(np.isin(unique_ref, [0, 1])):
            raise ValueError(f"Ground truth masks must be binary (0, 1), got values: {unique_ref}")

        pred = (pred > 0).astype(np.uint8)

        for i, (r, p) in enumerate(zip(ref, pred)):
            if ori_size is not None:
                ori_H, ori_W = ori_size[i]
                p = p[:ori_H, :ori_W]
                r = r[:ori_H, :ori_W]
            self.confusion_matrix += self._fast_hist(r.flatten(), p.flatten())

    def get_scores(self, binary=False, withbg=True):
        """
        Compute binary IoU from confusion matrix.

        Args:
            binary: Ignored, always True
            withbg: If True, return [bg_iou, fg_iou]; else, [fg_iou]

        Returns:
            iou: np.ndarray, IoU values
            mean_iou: float, Mean IoU
        """
        hist = self.confusion_matrix
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-7)
        if not withbg:
            iou = iou[1:]
        mean_iou = np.nanmean(iou)
        return iou, mean_iou

    def reset(self):
        self.confusion_matrix = np.zeros((2, 2))


class Accumulator(object):
    def __init__(self, **kwargs):
        self.values = kwargs
        self.counter = {k: 0 for k, v in kwargs.items()}
        for k, v in self.values.items():
            if not isinstance(v, (float, int, list)):
                raise TypeError(f"The Accumulator does not support `{type(v)}`. Supported types: [float, int, list]")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(self.values[k], list):
                self.values[k].append(v)
            else:
                self.values[k] = self.values[k] + v
            self.counter[k] += 1

    def reset(self):
        for k in self.values.keys():
            if isinstance(self.values[k], list):
                self.values[k] = []
            else:
                self.values[k] = 0
            self.counter[k] = 0

    def mean(self, key, axis=None, dic=False):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).mean(axis)
            else:
                return self.values[key] / (self.counter[key] + 1e-7)
        elif isinstance(key, (list, tuple)):
            if dic:
                return {k: self.mean(k, axis) for k in key}
            return [self.mean(k, axis) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")

    def std(self, key, axis=None, dic=False):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).std(axis)
            else:
                raise RuntimeError("`std` is not supported for (int, float). Use list instead.")
        elif isinstance(key, (list, tuple)):
            if dic:
                return {k: self.std(k) for k in key}
            return [self.std(k) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")

    def get(self):
        return {k: v.copy() if isinstance(v, list) else v for k, v in self.values.items()}
    
    def load(self, data):
        for k, v in data.items():
            if k in self.values:
                self.values[k] = v.copy() if isinstance(v, list) else v
                self.counter[k] = len(v) if isinstance(v, list) else 0
