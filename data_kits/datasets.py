import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import data_dir, lists_dir
from data_kits import transformation as tf
from data_kits import voc_coco as pfe
from utils_.misc import load_image
from data_kits.binary_dataset import BinarySegmentationDataset
import data_kits.transformation as tf
from constants import MEAN, STD


DATA_DIR = {
    "PASCAL": data_dir / "VOCdevkit/VOC2012",
    "COCO": data_dir / "COCO",
    "VISION24": data_dir / "VISION24",
     
}
DATA_LIST = {
    "PASCAL": {
        "train": lists_dir / "pascal/voc_sbd_merge_noduplicate.txt",
        "test": lists_dir / "pascal/val.txt",
        "eval_online": lists_dir / "pascal/val.txt"
    },
    "COCO": {
        "train": lists_dir / "coco/train_data_list.txt",
        "test": lists_dir / "coco/val_data_list.txt",
        "eval_online": lists_dir / "coco/val_data_list.txt"
    },
    "VISION24": {
        "train": lists_dir / "vision24/train.txt",
        "test": lists_dir / "vision24/test.txt",
        "eval_online": lists_dir / "vision24/test.txt"
    },
    
}


MEAN = [0.485, 0.456, 0.406]    # list, normalization mean in data preprocessing
STD = [0.229, 0.224, 0.225]     # list, normalization std in data preprocessing


def get_train_transforms(opt, height, width):
    """
    Exact implementation of Algorithm 1 from the Dinnar report.
    """
    supp_transform = tf.Compose([
        # 1) Deterministic resize to (height, width)
        tf.Resize(height, width),
        # 2) ColorJitter with 50% chance
        tf.ColorJitter(brightness=0.4, contrast=0.4,
                       saturation=0.4, hue=0.1, p=0.5),
        # 3) Grayscale with 20% chance
        tf.RandomGrayscale(p=0.2),
        # 4) GaussianBlur with 50% chance
        tf.RandomGaussianBlur(kernel_size=3, sigma=0.5, p=0.5),
        # 5) ToTensor & Normalize
        tf.ToTensor(mask_dtype='float'),
        tf.Normalize(MEAN, STD),
    ], processer=opt.proc)

    query_transform = tf.Compose([
        tf.Resize(height, width),
        tf.ColorJitter(brightness=0.4, contrast=0.4,
                       saturation=0.4, hue=0.1, p=0.5),
        tf.RandomGrayscale(p=0.2),
        tf.RandomGaussianBlur(kernel_size=3, sigma=0.5, p=0.5),
        tf.ToTensor(mask_dtype='long'),
        tf.Normalize(MEAN, STD),
    ], processer=opt.proc)

    return supp_transform, query_transform



def get_val_transforms(opt, height, width):
    supp_transform = tf.Compose([tf.Resize(height, width),
                                 tf.ToTensor(mask_dtype='float'),   # support mask using float
                                 tf.Normalize(MEAN, STD)], processer=opt.proc)

    query_transform = tf.Compose([tf.Resize(height, width, do_mask=False),  # keep mask the original size
                                  tf.ToTensor(mask_dtype='long'),   # query mask using long
                                  tf.Normalize(MEAN, STD)], processer=opt.proc)

    return supp_transform, query_transform


def load(opt, logger, mode):
    split, shot, query = opt.split, opt.shot, 1
    height, width = opt.height, opt.width

    if mode == "train":
        data_transform = get_train_transforms(opt, height, width)
    elif mode in ["test", "eval_online", "predict"]:
        data_transform = get_val_transforms(opt, height, width)
    else:
        raise ValueError(f'Not supported mode: {mode}. [train|eval_online|test|predict]')

    if opt.dataset == "PASCAL":
        num_classes = 20
        cache = True
    elif opt.dataset == "COCO":
        num_classes = 80
        cache = False
    elif opt.dataset == "VISION24":
        num_classes = 12
        cache = False
    else:
        raise ValueError(f'Not supported dataset: {opt.dataset}. [PASCAL|COCO | VISION24]')

    dataset = pfe.SemData(opt, split, shot, query,
                          data_root=DATA_DIR[opt.dataset],
                          data_list=DATA_LIST[opt.dataset][mode],
                          transform=data_transform,
                          mode=mode,
                          cache=cache)

    dataloader = DataLoader(dataset,
                            batch_size=opt.bs if mode == 'train' else opt.test_bs,
                            shuffle=True if mode == 'train' else False,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=True if mode == 'train' else False )

    logger.info(' ' * 5 + f"==> Data loader {opt.dataset} for {mode}")
    return dataset, dataloader, num_classes


def get_val_labels(opt, mode):
    if opt.dataset == "PASCAL":
        if opt.coco2pascal:
            if opt.split == 0:
                sub_val_list = [1, 4, 9, 11, 12, 15]
            elif opt.split == 1:
                sub_val_list = [2, 6, 13, 18]
            elif opt.split == 2:
                sub_val_list = [3, 7, 16, 17, 19, 20]
            elif opt.split == 3:
                sub_val_list = [5, 8, 10, 14]
            else:
                raise ValueError(f'PASCAL only have 4 splits [0|1|2|3], got {opt.split}')
        else:
            sub_val_list = list(range(opt.split * 5 + 1, opt.split * 5 + 6))
        return sub_val_list
    elif opt.dataset == "COCO":
        if opt.use_split_coco:
            return list(range(opt.split + 1, 81, 4))
        return list(range(opt.split * 20 + 1, opt.split * 20 + 21))
    elif opt.dataset == "VISION24":
        # return list(range(opt.split + 1, 6))
        return [i for i in range(0 , 13)]
    else:
        raise ValueError(f'Only support datasets [PASCAL|COCO|VISION24], got {opt.dataset}')


def load_p(opt, device):
    supp_t, query_t = get_val_transforms(opt, opt.height, opt.width)
    p = opt.p

    if p.sup and p.qry:
        supp_rgb_path = DATA_DIR[opt.dataset] / "JPEGImages" / f"{p.sup}.jpg"
        supp_lab_path = DATA_DIR[opt.dataset] / "SegmentationClassAug" / f"{p.sup}.png"
        query_rgb_path = DATA_DIR[opt.dataset] / "JPEGImages" / f"{p.qry}.jpg"
        query_lab_path = DATA_DIR[opt.dataset] / "SegmentationClassAug" / f"{p.qry}.png"

        supp_rgb = load_image(supp_rgb_path, 'img', opt.proc)
        _supp_lab = load_image(supp_lab_path, 'lab', opt.proc)
        supp_lab = np.zeros_like(_supp_lab, dtype=_supp_lab.dtype)
        supp_lab[_supp_lab == 255] = 255
        supp_lab[_supp_lab == p.cls] = 1
        query_ori = query_rgb = load_image(query_rgb_path, 'img', opt.proc)
        query_lab = np.zeros(query_rgb.shape[:-1], dtype=_supp_lab.dtype)
        _query_lab = load_image(query_lab_path, 'lab', opt.proc)
        query_lab[_query_lab == 255] = 255
        query_lab[_query_lab == p.cls] = 1

        supp_img, supp_lab, _ = supp_t(supp_rgb, supp_lab)
        query_img, query_lab, _ = query_t(query_rgb, query_lab)

        supp_img = supp_img[None, None].to(device)      # [B, S, 3, H, W]
        supp_lab = supp_lab[None, None].to(device)      # [B, S, H, W]
        query_img = query_img[None].to(device)          # [B, 3, H, W]
        query_lab = query_lab[None].to(device)      # [B, H, W]
    elif p.sup_rgb and p.sup_msk and p.qry_rgb:
        _supp_rgbs = [load_image(x, 'img', opt.proc) for x in p.sup_rgb]
        _supp_labs = [load_image(x, 'lab', opt.proc, mode=cv2.IMREAD_UNCHANGED) for x in p.sup_msk]
        supp_rgbs = []
        supp_labs = []
        for i, _supp_lab in enumerate(_supp_labs):
            if len(_supp_lab.shape) != 2:
                _supp_lab = _supp_lab[:, :, -1]
            supp_lab = np.zeros_like(_supp_lab, dtype=_supp_lab.dtype)
            if p.cls == 255:
                supp_lab[_supp_lab == p.cls] = 1
            else:
                supp_lab[_supp_lab == 255] = 255
                supp_lab[_supp_lab == p.cls] = 1
            supp_img, supp_lab, _ = supp_t(_supp_rgbs[i], supp_lab)
            supp_rgbs.append(supp_img)
            supp_labs.append(supp_lab)
        supp_img = torch.stack(supp_rgbs, dim=0)
        supp_lab = torch.stack(supp_labs, dim=0)

        query_ori = [load_image(x, 'img', opt.proc) for x in p.qry_rgb]
        _query_rgbs = query_ori
        _query_labs = [np.zeros(x.shape[:-1], dtype=_supp_labs[0].dtype) for x in _query_rgbs]
        query_rgbs = []
        for i, _query_lab in enumerate(_query_labs):
            query_img, query_lab, _ = query_t(_query_rgbs[i], _query_lab)
            query_rgbs.append(query_img)
        query_img = torch.stack(query_rgbs, dim=0)

        supp_img = supp_img[None].to(device)            # [B, S, 3, H, W]
        supp_lab = supp_lab[None].to(device)            # [B, S, H, W]
        query_img = query_img.to(device)                # [Q, 3, H, W]
        query_lab = None
    else:
        raise ValueError(f'In the prediction mode, either the [p.sup, p.qry] or the \n'
                         f'[p.sup_rgb, p.sup_msk, p.qry_rgb] should be given. Got \n'
                         f'    p.sup={p.sup}, p.qry={p.qry}\n'
                         f'    p.sup_rgb={p.sup_rgb}, p.sup_msk={p.sup_msk}, p.qry_rgb={p.qry_rgb}')


    return supp_img, supp_lab, query_img, query_lab, query_ori
