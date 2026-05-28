
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset
import clip



def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def yxyx_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    
    ymin, xmin, ymax, xmax = bbox
    return float(xmin), float(ymin), float(xmax), float(ymax)


def clamp_xyxy(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0.0, min(x1, img_w))
    x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h))
    y2 = max(0.0, min(y2, img_h))
    return x1, y1, x2, y2


def xyxy_to_xywh(x1, y1, x2, y2):
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


# -----------------------------
# 数据集
# -----------------------------
class VRDRelationDataset(Dataset):


    def __init__(
        self,
        ann_file: str,
        img_dir_root: str,
        clip_model_type: str = "ViT-B/32",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        objects2id_path: Optional[str] = None,
        predicate2id_path: Optional[str] = None,
        objects2id: Optional[Dict[str, int]] = None,
        predicate2id: Optional[Dict[str, int]] = None,
        fix_invalid_bbox: bool = False,
        on_missing: str = "skip",         
        index_subdirs: bool = True       
    ):
        super().__init__()
        self.ann_file = ann_file
        self.img_dir_root = img_dir_root
        self.anns: Dict[str, List[dict]] = load_json(ann_file) 
        self.image_names: List[str] = list(self.anns.keys())

        if objects2id is not None:
            self.objects2id = objects2id
        elif objects2id_path is not None:
            self.objects2id = load_json(objects2id_path)
        else:
            possible = os.path.join(os.path.dirname(ann_file), 'objects2id.json')
            self.objects2id = load_json(possible) if os.path.exists(possible) else {}

        if predicate2id is not None:
            self.predicate2id = predicate2id
        elif predicate2id_path is not None:
            self.predicate2id = load_json(predicate2id_path)
        else:
            possible = os.path.join(os.path.dirname(ann_file), 'predicate2id.json')
            self.predicate2id = load_json(possible) if os.path.exists(possible) else {}

        try:
            self.id2obj = {int(v): k for k, v in self.objects2id.items()}
        except Exception:
            self.id2obj = {int(k): v for k, v in self.objects2id.items()}

        try:
            self.id2pred = {int(v): k for k, v in self.predicate2id.items()}
        except Exception:
            self.id2pred = {int(k): v for k, v in self.predicate2id.items()}

       
        self.device = device
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.fix_invalid_bbox = fix_invalid_bbox
        self.on_missing = on_missing


        self._stem2path: Dict[str, str] = {}
        exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if index_subdirs:
            for root, _, files in os.walk(self.img_dir_root):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in exts:
                        stem = os.path.splitext(f)[0].lower()
                        self._stem2path[stem] = os.path.join(root, f)
        else:
            for f in os.listdir(self.img_dir_root):
                p = os.path.join(self.img_dir_root, f)
                if os.path.isfile(p):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in exts:
                        stem = os.path.splitext(f)[0].lower()
                        self._stem2path[stem] = p

    
        self.clip_model, self.clip_preprocess = clip.load(clip_model_type, device=device)
        self.clip_model.eval()

        self.vis_dim = 512
        self.text_dim = 512
        try:
            _device = device if device is not None else "cpu"
            dummy_img = self.clip_preprocess(Image.new('RGB', (224, 224))).unsqueeze(0).to(_device)
            with torch.no_grad():
                out = self.clip_model.encode_image(dummy_img)
            self.vis_dim = int(out.shape[-1])
            with torch.no_grad():
                tokens = clip.tokenize(["a"]).to(_device)
                tout = self.clip_model.encode_text(tokens)
            self.text_dim = int(tout.shape[-1])
        except Exception:
            pass

        self._cache_meta = {
            "clip": clip_model_type,
            "vis_dim": self.vis_dim,
            "text_dim": self.text_dim,
            "obj_size": len(self.id2obj),
            "pred_size": len(self.id2pred),
        }

  
    def _resolve_image_path(self, image_name: str) -> Optional[str]:
       
        p = os.path.join(self.img_dir_root, image_name)
        if os.path.exists(p):
            return p
        
        stem = os.path.splitext(os.path.basename(image_name))[0].lower()
        if stem in self._stem2path and os.path.exists(self._stem2path[stem]):
            return self._stem2path[stem]
        
        for ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'):
            cand = os.path.join(self.img_dir_root, stem + ext)
            if os.path.exists(cand):
                return cand
        return None

    def _load_image(self, image_name: str) -> Image.Image:
        p = self._resolve_image_path(image_name)
        if p is None:
            raise FileNotFoundError(f"Image not found for {image_name} under {self.img_dir_root}")
        try:
            img = Image.open(p)
            if getattr(img, "is_animated", False):
                try:
                    img.seek(0)  
                except Exception:
                    pass
            img = img.convert('RGB')
            return img
        except UnidentifiedImageError as e:
            raise FileNotFoundError(f"Unidentified or corrupt image file: {p}") from e

    @staticmethod
    def try_fix_bbox_yxyx(bbox: List[float]) -> Optional[List[float]]:
        ymin, xmin, ymax, xmax = bbox
        if ymax >= ymin and xmax >= xmin:
            return [ymin, xmin, ymax, xmax]
        
        if ymax > 0 and xmax > 0:
            cand = [ymin, xmin, ymin + ymax, xmin + xmax]
            if cand[2] >= cand[0] and cand[3] >= cand[1]:
                return cand
        
        if ymax < ymin and xmax >= xmin:
            cand = [ymax, xmin, ymin, xmax]
            if cand[2] >= cand[0]:
                return cand
        if xmax < xmin and ymax >= ymin:
            cand = [ymin, xmax, ymax, xmin]
            if cand[3] >= cand[1]:
                return cand
        if xmax < xmin and ymax < ymin:
            cand = [ymax, xmax, ymin, xmin]
            if cand[2] >= cand[0] and cand[3] >= cand[1]:
                return cand
        return None

   
    def _crop_and_preprocess(self, pil_img: Image.Image, box_xyxy: Tuple[float, float, float, float]):
        x1, y1, x2, y2 = box_xyxy
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if x2i <= x1i or y2i <= y1i:
            raise ValueError("degenerate crop")
        crop = pil_img.crop((x1i, y1i, x2i, y2i))
        return self.clip_preprocess(crop)

    def _encode_images_batch(self, crops_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.clip_model.encode_image(crops_tensor.to(self.device)).float().cpu()
        return feats

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            feats = self.clip_model.encode_text(tokens).float().cpu()
        return feats

    def _compute_global_image_feat(self, pil_img: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            img_t = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
            feat = self.clip_model.encode_image(img_t).float().squeeze(0).cpu()
        return feat

   
    def __len__(self) -> int:
        return len(self.image_names)

    def _empty_sample(self, image_name: str) -> Dict[str, Any]:
        return {
            'image_id': image_name,
            'image': torch.zeros(self.vis_dim),
            'sub_visual_feats': torch.zeros((0, self.vis_dim)),
            'obj_visual_feats': torch.zeros((0, self.vis_dim)),
            'union_visual_feats': torch.zeros((0, self.vis_dim)),
            'sub_text_feats': torch.zeros((0, self.text_dim)),
            'obj_text_feats': torch.zeros((0, self.text_dim)),
            'phrases': [],
            'sub_boxes': torch.zeros((0, 4)),
            'obj_boxes': torch.zeros((0, 4)),
            'gt_triplets': [],
            'labels': torch.zeros((0,), dtype=torch.long)
        }

    def _cache_path(self, image_name: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        stem = os.path.splitext(os.path.basename(image_name))[0]
        
        safe_clip = str(self._cache_meta['clip']).replace('/', '_')
        fname = f"{stem}__{safe_clip}.pt"
        return os.path.join(self.cache_dir, fname)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_name = self.image_names[idx]

        
        cpath = self._cache_path(image_name)
        if cpath and os.path.exists(cpath):
            try:
                sample = torch.load(cpath, map_location='cpu')
                
                ok = isinstance(sample, dict) \
                     and sample.get('image_id', None) == image_name \
                     and isinstance(sample.get('image', None), torch.Tensor) \
                     and isinstance(sample.get('labels', None), torch.Tensor)
                if ok:
                    return sample
            except Exception:
                
                pass

        
        try:
            pil_img = self._load_image(image_name)
        except FileNotFoundError:
            if self.on_missing == "skip":
                return self._empty_sample(image_name)
            else:
                raise

        relations: List[dict] = self.anns[image_name]
        img_w, img_h = pil_img.size

        sub_crops, obj_crops, union_crops = [], [], []
        sub_texts, obj_texts, phrases = [], [], []
        sub_boxes_xywh, obj_boxes_xywh = [], []
        labels, gt_triplets = [], []

        
        for rel in relations:
            subj = rel['subject']
            obj = rel['object']
            pred_id = int(rel['predicate'])

            sub_bbox = subj['bbox'][:]  # [ymin,xmin,ymax,xmax]
            obj_bbox = obj['bbox'][:]

            def valid(b):
                return (b[2] >= b[0]) and (b[3] >= b[1])

            if not valid(sub_bbox):
                if self.fix_invalid_bbox:
                    fixed = self.try_fix_bbox_yxyx(sub_bbox)
                    if fixed is None:
                        continue
                    sub_bbox = fixed
                else:
                    continue

            if not valid(obj_bbox):
                if self.fix_invalid_bbox:
                    fixed = self.try_fix_bbox_yxyx(obj_bbox)
                    if fixed is None:
                        continue
                    obj_bbox = fixed
                else:
                    continue

           
            sub_x1, sub_y1, sub_x2, sub_y2 = yxyx_to_xyxy(sub_bbox)
            obj_x1, obj_y1, obj_x2, obj_y2 = yxyx_to_xyxy(obj_bbox)
            sub_x1, sub_y1, sub_x2, sub_y2 = clamp_xyxy(sub_x1, sub_y1, sub_x2, sub_y2, img_w, img_h)
            obj_x1, obj_y1, obj_x2, obj_y2 = clamp_xyxy(obj_x1, obj_y1, obj_x2, obj_y2, img_w, img_h)

           
            union_x1 = min(sub_x1, obj_x1)
            union_y1 = min(sub_y1, obj_y1)
            union_x2 = max(sub_x2, obj_x2)
            union_y2 = max(sub_y2, obj_y2)
            union_x1, union_y1, union_x2, union_y2 = clamp_xyxy(union_x1, union_y1, union_x2, union_y2, img_w, img_h)

           
            sub_xywh = xyxy_to_xywh(sub_x1, sub_y1, sub_x2, sub_y2)
            obj_xywh = xyxy_to_xywh(obj_x1, obj_y1, obj_x2, obj_y2)

            
            try:
                sub_crop_t = self._crop_and_preprocess(pil_img, (sub_x1, sub_y1, sub_x2, sub_y2))
                obj_crop_t = self._crop_and_preprocess(pil_img, (obj_x1, obj_y1, obj_x2, obj_y2))
                union_crop_t = self._crop_and_preprocess(pil_img, (union_x1, union_y1, union_x2, union_y2))
            except Exception:
                
                continue

            sub_crops.append(sub_crop_t)
            obj_crops.append(obj_crop_t)
            union_crops.append(union_crop_t)

            sub_boxes_xywh.append(sub_xywh)
            obj_boxes_xywh.append(obj_xywh)

            sub_cls_id = int(subj['category'])
            obj_cls_id = int(obj['category'])
            sub_name = self.id2obj.get(sub_cls_id, str(sub_cls_id))
            obj_name = self.id2obj.get(obj_cls_id, str(obj_cls_id))
            pred_name = self.id2pred.get(pred_id, str(pred_id))

            sub_texts.append(sub_name)
            obj_texts.append(obj_name)
            phrases.append(f"{sub_name} {pred_name} {obj_name}")

            gt_triplets.append({
                'sub_cls': sub_cls_id,
                'obj_cls': obj_cls_id,
                'sub_box': sub_xywh,
                'obj_box': obj_xywh
            })

            labels.append(pred_id)

        N = len(sub_crops)
        if N == 0:
            
            return self._empty_sample(image_name)

        
        sub_batch = torch.stack(sub_crops, dim=0).to(self.device)
        obj_batch = torch.stack(obj_crops, dim=0).to(self.device)
        union_batch = torch.stack(union_crops, dim=0).to(self.device)

        with torch.no_grad():
            sub_feats = self._encode_images_batch(sub_batch)    # (N, vis_dim) on CPU
            obj_feats = self._encode_images_batch(obj_batch)    # (N, vis_dim)
            union_feats = self._encode_images_batch(union_batch)  # (N, vis_dim)

            sub_text_feats = self._encode_texts(sub_texts)      # (N, text_dim)
            obj_text_feats = self._encode_texts(obj_texts)      # (N, text_dim)
            global_img_feat = self._compute_global_image_feat(pil_img)  # (vis_dim,)

        sample = {
            'image_id': image_name,
            'image': global_img_feat,
            'sub_visual_feats': sub_feats,
            'obj_visual_feats': obj_feats,
            'union_visual_feats': union_feats,
            'sub_text_feats': sub_text_feats,
            'obj_text_feats': obj_text_feats,
            'phrases': phrases,
            'sub_boxes': torch.tensor(sub_boxes_xywh, dtype=torch.float32),
            'obj_boxes': torch.tensor(obj_boxes_xywh, dtype=torch.float32),
            'gt_triplets': gt_triplets,
            'labels': torch.tensor(labels, dtype=torch.long)
        }

        
        if cpath:
            try:
              
                tmp_path = cpath + ".tmp"
                torch.save(sample, tmp_path)
                os.replace(tmp_path, cpath)
            except Exception:
                
                pass

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
   
    batch = [b for b in batch if isinstance(b.get('labels', None), torch.Tensor) and b['labels'].numel() > 0]
    if len(batch) == 0:
        return {}
    keys = batch[0].keys()
    out: Dict[str, Any] = {k: [b[k] for b in batch] for k in keys}
    return out
