# %load core/metrics.py
import numpy as np
from pathlib import Path
from PIL import Image

class FewShotMetric(object):
    def __init__(self, n_class):
        self.n_class = n_class  # Number of defect classes (e.g., 12 for 0 to 11)
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
                # print(f"Sample {i} IoU: {iou_binary:.4f}")

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

def compute_metrics(pred_dir="cyclic_results/confident_predictions", gt_dir="/kaggle/working/updated_kaggle/data/VISION24/SegmentationClassAug", test_list="lists/vision24/test.txt", n_class=12, verbose=0, output_file="metrics_results.txt"):
    """
    Compute IoU, catch rate, yield rate, and mIoU for predicted masks against ground truth masks, using class labels from test.txt.

    Args:
        pred_dir (str): Path to the folder containing predicted masks (default: 'cyclic_results/filtered_predictions')
        gt_dir (str): Path to the folder containing ground truth masks (default: '/kaggle/working/updated_kaggle/data/VISION24/SegmentationClassAug')
        test_list (str): Path to test.txt containing image names and class labels (default: '/content/updated_kaggle/lists/vision24/test.txt')
        n_class (int): Number of classes for per-class IoU (default: 12)
        verbose (int): If > 0, print IoU for each image
        output_file (str): Path to save the metrics results

    Returns:
        dict: Dictionary containing mIoU, binary IoU, catch rate, yield rate, and per-class IoU
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    test_list = Path(test_list)

    # Define class mapping
    vision24_classes = [
        "Cable_thunderbolt", "Cable_torn_apart", "Cylinder_Chip", "Cylinder_Porosity", "Cylinder_RCS",
        "PCB_mouse_bite", "PCB_open_circuit", "PCB_short", "PCB_spur", "PCB_spurious_copper",
        "Screw_front", "Wood_impurities"
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(vision24_classes)}

    # Load test.txt to map image names to class indices
    cls_dict = {}
    if test_list.exists():
        with open(test_list, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue  # Skip malformed lines
                img_path, gt_path, class_name = parts
                # Extract base name (e.g., '001035_patch_0_0' from 'SegmentationClassAug/001035_patch_0_0.png')
                base_name = Path(gt_path).stem
                if class_name in class_to_idx:
                    cls_dict[base_name] = class_to_idx[class_name]
                else:
                    print(f"Warning: Class {class_name} not found in vision24_classes for {base_name}")
    else:
        raise FileNotFoundError(f"Test list file {test_list} not found")

    # Initialize metric
    metric = FewShotMetric(n_class=n_class)
    # Before the error occurs

    # Get list of predicted mask files
    pred_files = sorted([f for f in pred_dir.iterdir() if "_confident" in f.name.lower()])
    print("Matched predicted files:", [f.name for f in pred_files], flush=True)
    if not pred_files:
        raise ValueError(f"No predicted masks found in {pred_dir}")

    # Process each predicted mask
    for pred_file in pred_files:
        # Extract base name (e.g., '001035_patch_0_0' from '001035_patch_0_0_filtered.png')
        base_name = pred_file.stem.replace("_confident", "")
        gt_file = gt_dir / f"{base_name}.png"

        if not gt_file.exists():
            print(f"Warning: Ground truth mask {gt_file} not found for prediction {pred_file}")
            continue


        # Load predicted and ground truth masks
        pred_img = np.array(Image.open(pred_file).convert("L"))
        gt_img = np.array(Image.open(gt_file).convert("L"))

        # Convert to binary (0, 1)
        pred_img = (pred_img > 0).astype(np.uint8)  # Assuming 0 or 255 in predicted masks
        gt_img = (gt_img > 0).astype(np.uint8)  # Assuming 0 or 255 in ground truth masks

        # Ensure masks are binary
        if not np.all(np.isin(np.unique(pred_img), [0, 1])):
            raise ValueError(f"Predicted mask {pred_file} contains non-binary values: {np.unique(pred_img)}")
        if not np.all(np.isin(np.unique(gt_img), [0, 1])):
            raise ValueError(f"Ground truth mask {gt_file} contains non-binary values: {np.unique(gt_img)}")

        # Reshape to [batch, H, W]
        pred_img = pred_img[np.newaxis, :, :]  # Add batch dimension
        gt_img = gt_img[np.newaxis, :, :]      # Add batch dimension

        # Get class index from cls_dict, default to 0 if not found
        cls_idx = cls_dict.get(base_name, 0)
        cls = np.array([cls_idx])

        # Update metrics
        metric.update(pred_img, gt_img, cls, ori_size=None, verbose=verbose)

    # Compute final metrics
    class_iou, binary_iou, catch_rate, yield_rate = metric.get_scores(labels=None)

    # Compute mIoU (mean IoU across classes)
    mIoU = np.nanmean(class_iou)

    # Prepare results
    results = {
        "mIoU": mIoU,
        "binary_iou": binary_iou,
        "catch_rate": catch_rate,
        "yield_rate": yield_rate,
        "class_iou": class_iou.tolist(),
        "class_names": vision24_classes
    }

    # Save results to file
    with open(output_file, "w") as f:
        f.write(f"mIoU: {mIoU:.4f}\n")
        f.write(f"Binary IoU: {binary_iou:.4f}\n")
        f.write(f"Catch Rate: {catch_rate:.4f}\n")
        f.write(f"Yield Rate: {yield_rate:.4f}\n")
        f.write("Per-class IoU:\n")
        for cls_name, iou in zip(vision24_classes, class_iou):
            f.write(f"  {cls_name}: {iou:.4f}\n")

    if verbose:
        print(f"mIoU: {mIoU:.4f}")
        print(f"Binary IoU: {binary_iou:.4f}")
        print(f"Catch Rate: {catch_rate:.4f}")
        print(f"Yield Rate: {yield_rate:.4f}")
        print("Per-class IoU:")
        for cls_name, iou in zip(vision24_classes, class_iou):
            print(f"  {cls_name}: {iou:.4f}")

    return results
