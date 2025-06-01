# %load  data_kits/voc_coco.py
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import numpy as np
from constants import project_dir, data_dir
from utils_.misc import load_image, load_weights

cache_image = {}
cache_label = {}
cache_weights = {}


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, coco2pascal=False):
    assert split in [0, 1, 2, 3]
    data_list = Path(data_list).relative_to(project_dir)
    if not data_list.is_file():
        raise RuntimeError(f"Image list file do not exist: {data_list}\n")
    print('begin')
    filepath = data_list.parent / f"{data_list.stem}_{split}{'_coco2pascal' if coco2pascal else ''}.pkl"
    if filepath.exists():
        # Try to load `image_label_list` and `sub_class_file_list` from cache
        # print(filepath)
        with filepath.open("rb") as f:
            image_label_list, sub_class_file_list = pickle.load(f)
            
            # print(image_label_list)
            # print('----------------------------------')

        if len(image_label_list) > 0:
            return image_label_list, sub_class_file_list

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2,
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}

    # For file name --> if vision24
    dataset_name = str(filepath).split('/')[-2]
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = data_root / line_split[0]
        label_name = data_root / line_split[1]
        
        label = cv2.imread(str(label_name), cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()
        
        if(dataset_name == "vision24"):
            #read the label from the file
            label_class = line_split[2]
            label_class = [SemData.class_names['vision24'].index(str(label_class))]
            

        
            


        # if 0 in label_class:
        #     label_class.remove(0)
        # if 255 in label_class:
        #     label_class.remove(255)

        # print('-----------------')
        # print(label_class)
        # print('---------------')
        
        #replace each value 255 with 1
        
        
        
        new_label_class = []
        raw_label_list = []
        if(dataset_name == "vision24"):  
            pass
            # sub_list = [255 if x == 1 else x for x  in sub_list]
            
        
        
        for c in label_class:
            
            if c in sub_list:
                # c = 255 if c == 1 else c
                raw_label_list.append(c)
                # check the area of the mask
                tmp_label = np.zeros_like(label)
                
                target_pix = np.where(label == c)
               
                tmp_label[target_pix[0], target_pix[1]] = 1
                if tmp_label.sum() >= 2 * 32 * 32:
                    new_label_class.append(c)


        label_class = new_label_class
        
        # print('label class')
        # print('--------------------')
        # print(label_class)
        # print('--------------------')
            
            
        if(dataset_name == "vision24"):  
            #read the label from the file
            label_class = line_split[2]
            label_class = [SemData.class_names['vision24'].index(str(label_class))]
            sub_list = [1 if x == 255 else x for x  in sub_list]
            raw_label_list = [1 if x ==255 else x for x in raw_label_list]
            label_class = [1 if x == 255 else x for x in label_class]
        
        
        item = (
            image_name.relative_to(data_dir),
            label_name.relative_to(data_dir),
            raw_label_list
        )
        if len(label_class) > 0:
            image_label_list.append(item)
            if(dataset_name != "vision24"):
                #if not vision24 
                for c in label_class:
                    if c in sub_list:
                        sub_class_file_list[c].append(item)
            else:
                for c in label_class:
                    sub_class_file_list[c].append(item)
    
    with filepath.open("wb") as f:
        pickle.dump((image_label_list, sub_class_file_list), f)
    # print('----------')
    # print(image_label_list)
    # print(sub_class_file_list)
    # print('-----------------')
    print(f"Checking image&label list done! There are {len(image_label_list)} images in split {split}.")
    # print('sub_class_file_list')
    # print(sub_class_file_list.keys())
    
    return image_label_list, sub_class_file_list


class SemData(Dataset):
    class_names = {
        "pascal": ["background",
                   "airplane", "bicycle", "bird", "boat", "bottle",
                   "bus", "car", "cat", "chair", "cow",
                   "dining table", "dog", "horse", "motorbike", "person",
                   "potted plant", "sheep", "sofa", "train", "tv"
                   ],
        "coco": ["background",
                 "person", "airplane", "boat", "parking meter", "dog", "elephant", "backpack",
                 "suitcase", "sports ball", "skateboard", "wine glass", "spoon", "sandwich", "hot dog",
                 "chair", "dining table", "mouse", "microwave", "refrigerator", "scissors",
                 "bicycle", "bus", "traffic light", "bench", "horse", "bear", "umbrella",
                 "frisbee", "kite", "surfboard", "cup", "bowl", "orange", "pizza",
                 "couch", "toilet", "remote", "oven", "book", "teddy bear",
                 "car", "train", "fire hydrant", "bird", "sheep", "zebra", "handbag",
                 "skis", "baseball bat", "tennis racket", "fork", "banana", "broccoli", "donut",
                 "potted plant", "tv", "keyboard", "toaster", "clock", "hair drier",
                 "motorcycle", "truck", "stop sign", "cat", "cow", "giraffe", "tie",
                 "snowboard", "baseball glove", "bottle", "knife", "apple", "carrot", "cake",
                 "bed", "laptop", "cell phone", "sink", "vase", "toothbrush",
                 ],
        # "vision24":["defect" , "non-defect"],
        "vision24" : ["Cable_thunderbolt" , "Cable_torn_apart" , "Cylinder_Chip" , "Cylinder_Porosity" ,"Cylinder_RCS",
                          "PCB_mouse_bite" , "PCB_open_circuit" , "PCB_short" , "PCB_spur" , "PCB_spurious_copper" , "Screw_front",
                          "Wood_impurities"]
 
        
    }

    def __init__(self, opt, split, shot, query,
                 data_root=None, data_list=None, transform=None, mode='train',
                 cache=True):
        assert mode in ['train', 'val', 'test', 'eval_online']

        if mode != "train" and opt.dataset in ["PASCAL", "COCO" , "VISION24"]:
            mode = "val"
        self.opt = opt
        self.mode = mode
        self.split = split
        self.shot = shot
        self.query = query
        self.data_root = Path(data_root)
        self.tasks = []
        self.transform = transform
        self.cache = cache

        if opt.dataset == "PASCAL":
            n_class = 20
            interval = 5
            self.class_list = list(range(1, 21))
            self.sub_val_list = list(range(interval * split + 1, interval * (split + 1) + 1))
            self.sub_list = list(set(range(1, n_class + 1)) - set(self.sub_val_list))
            
            if opt.coco2pascal:
                n_class = 20
                self.class_list = list(range(1, 21))
                if split == 0:
                    self.sub_val_list = [1, 4, 9, 11, 12, 15]
                elif split == 1:
                    self.sub_val_list = [2, 6, 13, 18]
                elif split == 2:
                    self.sub_val_list = [3, 7, 16, 17, 19, 20]
                elif split == 3:
                    self.sub_val_list = [5, 8, 10, 14]
                self.sub_list = list(set(range(1, 21)) - set(self.sub_val_list))

        elif opt.dataset == "COCO":
            n_class = 80
            if opt.use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                self.sub_val_list = list(range(split + 1, 81, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            else:
                interval = 20
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                self.sub_val_list = list(range(interval * split + 1, interval * (split + 1) + 1))
                self.sub_list = list(set(range(1, n_class + 1)) - set(self.sub_val_list))
        elif opt.dataset == "VISION24":
            n_class =12;
            interval = 2
            # self.class_list =[ i for i in range(len(self.class_names['vision24']))] 
            # self.sub_val_list =  [ i for i in range(len(self.class_names['vision24']))]
            # self.sub_list = [ i for i in range(len(self.class_names['vision24']))]

            self.class_list = [i for i in range(n_class)]
            self.sub_val_list = [i for i in range(n_class)]
            self.sub_list = [i for i in range(n_class)]
            
            # print('1--------------------------------')
            # print(self.sub_val_list)
            # print('\n\n')
            # print('2--------------------------------')
            # print(self.sub_list)
        else:
            raise ValueError(f'Not supported dataset: {opt.dataset}. [PASCAL|COCO | VISION24]')

        self.n_class = n_class
        self.train_num_classes = len(self.sub_list)
        if self.mode == 'train':
            self.data_list, self.sub_cls_files = make_dataset(
                split, self.data_root, data_list, self.sub_list)
            
        else:
            self.data_list, self.sub_cls_files = make_dataset(
                split, self.data_root, data_list, self.sub_val_list, opt.coco2pascal)

        # print('----------------------------------')
        # print(self.data_list)
        # print(type(self.data_list))
        # print('-------------------------')
        for c in self.sub_cls_files:
            print(f"Class {c}: {len(self.sub_cls_files[c])} images")

        self.length_data_list = len(self.data_list)
        print(f"Data list length: {self.length_data_list}")
        assert self.length_data_list > 0, 'Length of data list is 0. Please make sure the data root ' \
                                          f'({self.data_root}) and data list ({data_list}) are correct.'
        self.length_sub_class_list = {k: len(v) for k, v in self.sub_cls_files.items()}

        self.reset_sampler()

    def __len__(self):
        if self.mode == 'train':
            return self.opt.train_n or self.length_data_list
        else:
            return self.opt.test_n or self.length_data_list

    def reset_sampler(self):
        seed = self.opt.seed
        test_seed = self.opt.test_seed
        # Use fixed test sampler(opt.test_seed) for reproducibility
        self.sampler = np.random.RandomState(seed) \
            if self.mode == "train" else np.random.RandomState(test_seed)

    def sample_tasks(self):
        self.tasks = []

        total_length = len(self)
        rounds = (total_length + self.length_data_list - 1) // self.length_data_list
        counter = 0
        support = []
        for r in range(rounds):
            # Sampling random episodes. Use self.reset_sampler() for fixed sampling orders
            rng = self.sampler.permutation(np.arange(self.length_data_list)) 
            for idx in rng:
                item = self.data_list[idx]
                
                assert len(item[2]) > 0

                cls = self.sampler.choice(item[2])
                
                num_files = len(self.sub_cls_files[cls])
                
               
                s_indices = []
                image_path = ""
                for i in range(self.shot):
                    while True:
                             
                        s_idx = self.sampler.choice(num_files, size=1)[0]
                        if self.sub_cls_files[cls][s_idx] == item or s_idx in s_indices:
                            continue
                        
                        image_path, label_path, _ = self.data_list[s_idx]
                        image = self.get_image(label_path)
                        

                        img_np = np.array(image)
                        if not (img_np == 255).any():
                            continue
                        else:
                            s_indices.append(s_idx)
                            support.append(s_idx)
                            break
                            
                        # for i in range(image.shape[0]):
                        #     p = image[i]
                        #     if(np.sum(p)==0):
                        #         pass
                        #     else:
                        #         has_white= True
                        #         break
                        # if(has_white):        
                        #     s_indices.append(s_idx)
                        #     support.append(s_idx)
                        # else:
                        #     continue
                    
                self.tasks.append((idx, cls, s_indices))
                counter += 1
                if counter >= total_length:
                    break
        with open('output.txt' , 'w') as wf:
            for x in support:
                wf.write(str(self.data_list[x][0]) + '\n')
        
        exit(1)
                
                
       
            
            

    def seg_encode(self, lab, cls, ignore_lab):
        if self.opt.proc == 'pil':
            lab = np.array(lab, np.uint8)
        target_pix = np.where(lab == cls)
        ignore_pix = np.where(lab == 255)
        lab[:, :] = 0
        if target_pix[0].shape[0] > 0:
            lab[target_pix[0], target_pix[1]] = 1
        if ignore_pix[0].shape[0] > 0:
            lab[ignore_pix[0], ignore_pix[1]] = ignore_lab
        if self.opt.proc == 'pil':
            lab = Image.fromarray(lab)
        return lab
    # def seg_encode(self, lab, cls, ignore_lab):
    #     if self.opt.proc == 'pil':
    #         lab = np.array(lab, np.uint8)
    
    #     target_pix = np.where(lab == 255)  # Find all pixels with 255
    #     background_pix = np.where(lab == 0)
    
    #     lab[:, :] = 0  # Set all pixels to background (0)
    #     if target_pix[0].shape[0] > 0:
    #         lab[target_pix[0], target_pix[1]] = 1  # Convert 255 → 1 for class index
    #     if ignore_pix[0].shape[0] > 0:
    #         lab[ignore_pix[0], ignore_pix[1]] = ignore_lab
    
    #     if self.opt.proc == 'pil':
    #         lab = Image.fromarray(lab)
    
    #     return lab


    def get_image(self, name, cache=True):
        if self.cache and cache:
            if name not in cache_image:
                cache_image[name] = load_image(data_dir / name, 'img', self.opt.proc)
            return cache_image[name].copy()
        else:
            return load_image(data_dir / name, 'img', self.opt.proc)

    def get_label(self, name, cls, ignore_lab=0, cache=True):
        if self.cache and cache:
            if name not in cache_label:
                cache_label[name] = load_image(data_dir / name, 'lab', self.opt.proc)
            lab = cache_label[name].copy()
        else:
            lab = load_image(data_dir / name, 'lab', self.opt.proc)
        lab = self.seg_encode(lab, cls, ignore_lab)
        # Convert 255 to 1
        
        lab = np.where(lab == 255, 1, 0).astype(np.uint8)
        return lab

    def get_weights(self, name, cls, cache=True):
        if self.cache and cache:
            if name not in cache_weights:
                cache_weights[name] = load_weights(data_dir / name)
            weights_dict = cache_weights[name]
        else:
            weights_dict = load_weights(data_dir / name)
        if cls in weights_dict['c']:
            class_index = weights_dict['c'].index(cls)
            weights = weights_dict['x'][class_index].copy().astype(np.float32)
        else:
            weights = np.zeros(weights_dict['x'][0].shape, np.float32)
        return weights
    
    # def get_weights(self, name, cls, cache=True):
    #     weight_path = data_dir / name
    
    #     # If file doesn't exist, return default zero weights
    #     if not os.path.exists(weight_path):
    #         print(f"Warning: Weight file {weight_path} not found. Using zero weights.")
    #         return np.zeros((1, 768), np.float32)  # Adjust shape as needed
    
    #     if self.cache and cache:
    #         if name not in cache_weights:
    #             cache_weights[name] = load_weights(weight_path)
    #         weights_dict = cache_weights[name]
    #     else:
    #         weights_dict = load_weights(weight_path)
    
    #     if cls in weights_dict['c']:
    #         class_index = weights_dict['c'].index(cls)
    #         weights = weights_dict['x'][class_index].copy().astype(np.float32)
    #     else:
    #         weights = np.zeros(weights_dict['x'][0].shape, np.float32)
    
    #     return weights

    def __getitem__(self, index):
        qry_idx, cls, support_indices = self.tasks[index]
        image_path, label_path, _ = self.data_list[qry_idx]
        sup_image_paths = []
        sup_label_paths = []
       
        for sup_idx in support_indices:
            x_path, y_path, _ = self.sub_cls_files[cls][sup_idx]
            sup_image_paths.append(x_path)
            sup_label_paths.append(y_path)

        
        qry_names = [image_path.stem]
        sup_names = [x.stem for x in sup_image_paths]

        image = self.get_image(image_path)
        # keep query ignore_label as 255, which has significant to iou.
        label = self.get_label(label_path, cls, ignore_lab=255)
        kwargs = {}
        sup_kwargs = [{} for _ in range(self.opt.shot)]
        # Load weights
        if self.mode == 'train':
            if self.opt.precompute_weight and self.opt.loss == 'cedt':
                # Load precomputed weights, used for computing loss
                weight_path = image_path.parents[1] / f"weights/{image_path.stem}.npz"
                kwargs['weights'] = self.get_weights(weight_path, cls)


        
        sup_images = [self.get_image(x) for x in sup_image_paths]
        sup_labels = [self.get_label(x, cls, ignore_lab=255) for x in sup_label_paths]

        # raw_label = label.copy()
        if self.transform is not None:
            image, label, kwargs = self.transform[1](image, label, **kwargs)
            for k in range(self.shot):
                sup_images[k], sup_labels[k], sup_kwargs[k] = self.transform[0](sup_images[k], sup_labels[k],
                                                                                **sup_kwargs[k])

        sup_images = torch.stack(sup_images, dim=0)
        sup_labels = torch.stack(sup_labels, dim=0)
        label = label.unsqueeze(dim=0)

        ret_dict = {
            'sup_rgb': sup_images,  # [S, 3, H, W]
            'sup_msk': sup_labels,  # [S, H, W]
            'qry_rgb': image,  # [3, H, W]
            'qry_msk': label,  # [1, H, W]
            'cls': cls,  # [], values in [1, 20] for PASCAL
            'weights': kwargs.get('weights', None),
            'sup_names': sup_names,
            'qry_names': qry_names,
        }
        ret_dict = {k: v for k, v in ret_dict.items() if v is not None}
        return ret_dict

    def get_class_name(self, cls, dataset):
        return self.class_names[dataset.lower()][cls]
