# %load run.py
# %load kaggle/working/FPTrans/run.py
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sacred import Experiment

from config import setup, init_environment
from constants import on_cloud
from core.base_trainer import BaseTrainer, BaseEvaluator
from core.losses import get as get_loss_obj
from data_kits import datasets
from networks import load_model
from utils_ import misc

ex = setup(
    Experiment(name="FPTrans", save_git_info=False, base_dir="./")
)
torch.set_printoptions(precision=8)

import numpy as np
from PIL import Image
from pathlib import Path
import torch

class Evaluator(BaseEvaluator):
    def test_step(self, batch, step):
        # Extract inputs from batch
        sup_rgb = batch['sup_rgb'].cuda()
        sup_msk = batch['sup_msk'].cuda()
        qry_rgb = batch['qry_rgb'].cuda()
        qry_msk = batch['qry_msk'].cuda()
        qry_name = batch['qry_names']  # Single query image name
        classes = batch['cls'].cuda()

        # Forward pass
        output = self.model_DP(qry_rgb, sup_rgb, sup_msk, qry_msk)
        qry_pred = output['out']

        # Compute loss
        loss = self.loss_obj(qry_pred, qry_msk.squeeze(1))

        # Compute prediction
        qry_pred = qry_pred.argmax(dim=1).detach().cpu().numpy()

        # Save binary predicted mask to file if output directory is specified
        if hasattr(self.opt, 'p') and hasattr(self.opt.p, 'out') and self.opt.p.out:
            # Create output directory if it doesn't exist
            out_dir = Path(self.opt.p.out)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Ensure qry_name is a string or a single-item list/tuple
            if isinstance(qry_name, (list, tuple)):
                if len(qry_name) != 1:
                    raise ValueError("Expected a single query name, got multiple: {qry_name}")
                qry_name = qry_name[0][0]
            if not isinstance(qry_name, str):
                raise ValueError("batch['qry_names'] must be a string or a single-item list/tuple of a string")

            # Process the single prediction
            pred = qry_pred[0].astype(np.uint8) * 255  # Binary mask (0 or 255)

            # Use query image name with _pred suffix
            out_name = Path(qry_name).stem + '_pred.png'
            out_path = out_dir / out_name

            # Save as grayscale binary mask
            Image.fromarray(pred).convert('L').save(out_path)

        # Clean up memory
        del output
        torch.cuda.empty_cache()

        return qry_pred, {'loss': loss.item()}




class CyclicConsistencyEvaluator(BaseEvaluator):
    def __init__(self, opt, logger, device, model, data_loader, eval_name="EVAL_CYCLIC", threshold=0.18):
        super().__init__(opt, logger, device, model, data_loader, eval_name)
        self.threshold = threshold
        self.model.eval()

    def get_accumulator_keys(self):
        return ['loss', 'confidence_score', 'forward_prob', 'reverse_prob', 'miou_score', 'is_confident']

    def compute_miou(self, pred_mask, gt_mask):
        pred_mask = pred_mask.astype(bool)
        gt_mask = gt_mask.astype(bool)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)

    def cyclic_consistency_check(self, sup_rgb, sup_msk, qry_rgb, forward_pred, forward_prob):
        """
        Perform cyclic consistency check.
        Args:
            sup_rgb: Support image [1, C, H, W]
            sup_msk: Support mask [1, 1, H, W]
            qry_rgb: Query image [1, C, H, W]
            forward_pred: Forward prediction mask [H, W]
            forward_prob: Forward prediction probability
        Returns:
            tuple: (confidence_score, reverse_pred, reverse_prob, miou_score)
        """
         # Convert forward prediction to tensor format for reverse inference
        forward_pred_tensor = torch.from_numpy(forward_pred).float().cuda()

        def ensure_5d(tensor, is_mask=False):
          if tensor.ndim == 4:
              if is_mask:
                  return tensor.unsqueeze(2)  # Add C dim for mask: [B, S, 1, H, W]
              else:
                  return tensor.unsqueeze(1)  # Add S dim: [B, 1, C, H, W]
          elif tensor.ndim == 3:
              return tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
          return tensor

        sup_rgb = ensure_5d(sup_rgb)
        qry_rgb = ensure_5d(qry_rgb)
        sup_msk = ensure_5d(sup_msk, is_mask=True)
        forward_pred_tensor = ensure_5d(forward_pred_tensor, is_mask=True)
        forward_pred_tensor = torch.from_numpy(forward_pred).float().cuda()
        if len(forward_pred_tensor.shape) == 2:
            forward_pred_tensor = forward_pred_tensor.unsqueeze(0).unsqueeze(0)
        elif len(forward_pred_tensor.shape) == 3:
            forward_pred_tensor = forward_pred_tensor.unsqueeze(0)

        reverse_output = self.model_DP(sup_rgb, qry_rgb, forward_pred_tensor, sup_msk)
        reverse_pred_logits = reverse_output['out']
        reverse_prob = torch.softmax(reverse_pred_logits, dim=1).max(dim=1)[0].mean().item()
        reverse_pred = reverse_pred_logits.argmax(dim=1).detach().cpu().numpy()[0]
        original_sup_mask = sup_msk.cpu().numpy()[0, 0]
        miou_score = self.compute_miou(reverse_pred, original_sup_mask)
        confidence_score = forward_prob * reverse_prob * miou_score



        return confidence_score, reverse_pred, reverse_prob, miou_score

    def test_step(self, batch, step):
        # Extract inputs
        sup_rgb = batch['sup_rgb'].to(self.device)  # [1, C, H, W]
        sup_msk = batch['sup_msk'].to(self.device)  # [1, 1, H, W]
        qry_rgb = batch['qry_rgb'].to(self.device)  # [1, C, H, W]
        qry_msk = batch['qry_msk'].to(self.device)  # [1, 1, H, W]
        qry_name = batch['qry_names']
        classes = batch['cls'].to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model_DP(qry_rgb, sup_rgb, sup_msk, qry_msk)
        qry_pred_logits = output['out']  # [1, num_classes, H, W]
        loss = self.loss_obj(qry_pred_logits, qry_msk.squeeze(1))
        forward_prob = torch.softmax(qry_pred_logits, dim=1).max(dim=1)[0].mean().item()
        qry_pred = qry_pred_logits.argmax(dim=1).detach().cpu().numpy()  # [1, H, W]

        # Cyclic consistency check
        confidence_score, reverse_pred, reverse_prob, miou_score = self.cyclic_consistency_check(
            sup_rgb, sup_msk, qry_rgb, qry_pred[0], forward_prob
        )

        # Filter predictions
        is_confident = confidence_score >= self.threshold
        filtered_pred = qry_pred.copy()
        if not is_confident:
            filtered_pred[0] = np.zeros_like(filtered_pred[0])

        # Save results
        if hasattr(self.opt, 'p') and hasattr(self.opt.p, 'out') and self.opt.p.out:
            self._save_cyclic_results(
                qry_name, qry_pred[0], filtered_pred[0], reverse_pred,
                confidence_score, forward_prob, reverse_prob, miou_score, is_confident
            )

        # Clean up
        del output
        torch.cuda.empty_cache()

        return filtered_pred, {
            'loss': loss.item(),
            'confidence_score': confidence_score,
            'forward_prob': forward_prob,
            'reverse_prob': reverse_prob,
            'miou_score': miou_score,
            'is_confident': float(is_confident)
        }

    def _save_cyclic_results(self, qry_name, original_pred, filtered_pred, reverse_pred,
                            confidence_score, forward_prob, reverse_prob, miou_score, is_confident):
        out_dir = Path(self.opt.p.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Handle qry_name
        if isinstance(qry_name, (list, tuple)):
            if len(qry_name) != 1:
                raise ValueError(f"Expected a single query name, got multiple: {qry_name}")
            qry_name = qry_name[0][0]
        if not isinstance(qry_name, str):
            raise ValueError("batch['qry_names'] must be a string or a single-item list/tuple of a string")

        # Create subdirectories
        for subdir in ["original_predictions", "filtered_predictions", "reverse_predictions",
                       "confident_predictions", "rejected_predictions"]:
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)

        base_name = Path(qry_name).stem

        # Save masks
        for mask, suffix, target_dir in [
            (original_pred, "original", "original_predictions"),
            (filtered_pred, "filtered", "filtered_predictions"),
            (reverse_pred, "reverse", "reverse_predictions"),
            (filtered_pred if is_confident else original_pred, "confident" if is_confident else "rejected",
             "confident_predictions" if is_confident else "rejected_predictions")
        ]:
            if mask is not None:
                mask_img = (mask.astype(np.uint8) * 255)
                Image.fromarray(mask_img).convert('L').save(out_dir / target_dir / f"{base_name}_{suffix}.png")

        # Save metadata
        with open(out_dir / "cyclic_consistency_results.txt", "a") as f:
            f.write(f"{base_name}: confidence={confidence_score:.4f}, forward_prob={forward_prob:.4f}, "
                    f"reverse_prob={reverse_prob:.4f}, miou={miou_score:.4f}, confident={is_confident}\n")
class Trainer(BaseTrainer):
    def _train_step(self, batch, step, epoch):
        sup_rgb = batch['sup_rgb'].cuda()
        sup_msk = batch['sup_msk'].cuda()
        qry_rgb = batch['qry_rgb'].cuda()
        qry_msk = batch['qry_msk'].cuda()
        classes = batch['cls'].cuda()
        kwargs = {}
        if 'weights' in batch:
            kwargs['weight'] = batch['weights'].cuda()

        output = self.model_DP(qry_rgb, sup_rgb, sup_msk, qry_msk)
        qry_msk_reshape = qry_msk.view(-1, *qry_msk.shape[-2:])

        loss = self.loss_obj(output['out'], qry_msk_reshape, **kwargs)
        loss_prompt = self.loss_obj(output['out_prompt'], qry_msk_reshape, **kwargs)
        if len(output['loss_pair'].shape) == 0:     # single GPU
            loss_pair = output['loss_pair']
        else:   # multiple GPUs
            loss_pair = output['loss_pair'].mean(0)
        loss_pair = loss_pair * self.opt.pair_lossW

        total_loss = loss + loss_prompt + loss_pair
        return total_loss, loss, loss_prompt, loss_pair

    def train_step(self, batch, step, epoch):
        self.optimizer.zero_grad()

        total_loss, loss, loss_prompt, loss_pair = self._train_step(batch, step, epoch)

        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'prompt': loss_prompt.item(),
            'pair': loss_pair.item(),
        }


@ex.main
def train(_run, _config):
    opt, logger, device = init_environment(ex, _run, _config)

    ds_train, data_loader, _ = datasets.load(opt, logger, "train")
    ds_eval_online, data_loader_val, num_classes = datasets.load(opt, logger, "eval_online")

    logger.info(f'     ==> {len(ds_train)} training samples')
    logger.info(f'     ==> {len(ds_eval_online)} eval_online samples')

    model = load_model(opt, logger)
    if opt.exp_id >= 0 or opt.ckpt:

        ckpt = misc.find_snapshot(_run.run_dir.parent, opt.exp_id, opt.ckpt, afs=on_cloud)
        model.load_weights(ckpt, logger, strict=opt.strict)

    trainer = Trainer(opt, logger, device, model, data_loader, data_loader_val, _run)
    evaluator = Evaluator(opt, logger, device, trainer.model_DP, None, "EVAL_ONLINE")

    logger.info("Start training.")
    start_epoch = 1
    trainer.start_training_loop(start_epoch, evaluator, num_classes)

    logger.info(f"============ Training finished - id {_run._id} ============\n")
    if _run._id is not None:
        return test(_run, _config, _run._id, ckpt=None, strict=False, eval_after_train=True)


@ex.command(unobserved=True)
def test(_run, _config, exp_id=-1, ckpt=None, strict=True, eval_after_train=False, use_cyclic=False, cyclic_threshold=0.18):
    opt, logger, device = init_environment(ex, _run, _config, eval_after_train=eval_after_train)

    ds_test, data_loader, num_classes = datasets.load(opt, logger, "test")

    logger.info(f'     ==> {len(ds_test)} testing samples')

    model = load_model(opt, logger)
    if not opt.no_resume:
        model_ckpt = misc.find_snapshot(_run.run_dir.parent, exp_id, ckpt)
        logger.info(f"     ==> Try to load checkpoint from {model_ckpt}")
        model.load_weights(model_ckpt, logger, strict=strict)
        logger.info(f"     ==> Checkpoint loaded.")

    # Choose evaluator based on cyclic consistency flag
    if use_cyclic:
        logger.info(f"Using Cyclic Consistency Evaluator with threshold {cyclic_threshold}")
        tester = CyclicConsistencyEvaluator(opt, logger, device, model, None, "EVAL_CYCLIC", threshold=cyclic_threshold)
    else:
        tester = Evaluator(opt, logger, device, model, None, "EVAL")

    logger.info("Start testing.")

    loss, mean_iou, binary_iou, _, _ , _,  _,  catch_rate, yeild_rate = tester.start_eval_loop(data_loader, num_classes)

    return f"Loss: {loss:.4f}, mIoU: {mean_iou * 100:.2f}, bIoU: {binary_iou * 100:.2f}, catch_rate:{catch_rate * 100:.2f}, yeild_rate: {yeild_rate * 100:.2f}"


@ex.command(unobserved=True)
def test_cyclic(_run, _config, exp_id=-1, ckpt=None, strict=True, threshold=0.18):
    """Test with cyclic consistency"""
    return test(_run, _config, exp_id, ckpt, strict, eval_after_train=False, use_cyclic=True, cyclic_threshold=threshold)


@ex.command(unobserved=True)
def predict(_run, _config, exp_id=-1, ckpt=None, strict=True):
    opt, logger, device = init_environment(ex, _run, _config)

    model = load_model(opt, logger)
    if not opt.no_resume:
        model_ckpt = misc.find_snapshot(_run.run_dir.parent, exp_id, ckpt)
        logger.info(f"     ==> Try to load checkpoint from {model_ckpt}")
        model.load_weights(model_ckpt, logger, strict)
        logger.info(f"     ==> Checkpoint loaded.")
    model = model.to(device)
    loss_obj = get_loss_obj(opt, logger, loss='ce')

    sup_rgb, sup_msk, qry_rgb, qry_msk, qry_ori = datasets.load_p(opt, device)
    classes = torch.LongTensor([opt.p.cls]).cuda()

    logger.info("Start predicting.")

    model.eval()
    ret_values = []
    for i in range(qry_rgb.shape[0]):
        print('Processing:', i + 1)
        qry_rgb_i = qry_rgb[i:i + 1]
        qry_msk_i = qry_msk[i:i + 1] if qry_msk is not None else None
        qry_ori_i = qry_ori[i]

        output = model(qry_rgb_i, sup_rgb, sup_msk, out_shape=qry_ori_i.shape[-3:-1])
        pred = output['out'].argmax(dim=1).detach().cpu().numpy()

        if qry_msk_i is not None:
            loss = loss_obj(output['out'], qry_msk_i).item()
            ref = qry_msk_i.cpu().numpy()
            tp = int((np.logical_and(pred == 1, ref != 255) * np.logical_and(ref == 1, ref != 255)).sum())
            fp = int((np.logical_and(pred == 1, ref != 255) * np.logical_and(ref != 1, ref != 255)).sum())
            fn = int((np.logical_and(pred != 1, ref != 255) * np.logical_and(ref == 1, ref != 255)).sum())
            mean_iou = tp / (tp + fp + fn)
            binary_iou = 0
            ret_values.append(f"Loss: {loss:.4f}, mIoU: {mean_iou * 100:.2f}, bIoU: {binary_iou * 100:.2f}")
        else:
            ret_values.append(None)

        # Save to file
        if opt.p.out:
            pred = pred[0].astype(np.uint8) * 255
            if opt.p.overlap:
                out = qry_ori_i.copy()
                # out[pred == 255] = out[pred == 255] * 0.5 + np.array([255, 0, 0]) * 0.5
                pred[pred == 255] = 1  # Ensure consistency
                out[pred == 1] = out[pred == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
            else:
                out = pred

            out_dir = Path(opt.p.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = Path(opt.p.qry or opt.p.qry_rgb[i]).stem + '_pred.png'
            out_path = out_dir / out_name
            Image.fromarray(out).save(out_path)

        # Release memory
        del output
        torch.cuda.empty_cache()

    if ret_values[0] is not None:
        return '\n'.join(ret_values)


if __name__ == '__main__':
    ex.run_commandline()
