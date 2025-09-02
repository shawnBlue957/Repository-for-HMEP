
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip
from feature_cache import FeatureCache

# predicate -> super-class mapping
predicate2superclass = {
    "above": "geometric",
    "across": "geometric",
    "against": "geometric",
    "along": "geometric",
    "and": "geometric",
    "at": "geometric",
    "behind": "geometric",
    "between": "geometric",
    "in": "geometric",
    "in front of": "geometric",
    "near": "geometric",
    "on": "geometric",
    "on back of": "geometric",
    "over": "geometric",
    "under": "geometric",
    "belonging to": "posession",
    "for": "posession",
    "from": "posession",
    "has": "posession",
    "made of": "posession",
    "of": "posession",
    "part of": "posession",
    "to": "posession",
    "wearing": "posession",
    "wears": "posession",
    "with": "posession",
    "attached to": "semantic",
    "carrying": "semantic",
    "covered in": "semantic",
    "covering": "semantic",
    "eating": "semantic",
    "flying in": "semantic",
    "growing on": "semantic",
    "hanging from": "semantic",
    "holding": "semantic",
    "laying on": "semantic",
    "looking at": "semantic",
    "lying on": "semantic",
    "mounted on": "semantic",
    "painted on": "semantic",
    "parked on": "semantic",
    "playing": "semantic",
    "riding": "semantic",
    "says": "semantic",
    "sitting on": "semantic",
    "standing on": "semantic",
    "using": "semantic",
    "walking in": "semantic",
    "walking on": "semantic",
    "watching": "semantic"
}


class VGRelationDataset(Dataset):
    def __init__(self,
                 ann_file,
                 img_dir_root,
                 object_json,
                 clip_model_type="ViT-B/32",
                 device="cpu",
                 cache_dir=None):
        super().__init__()
        self.device = device
        self.img_dir_root = img_dir_root

        with open(ann_file, 'r') as f:
            ann = json.load(f)
        self.images = {img["id"]: img for img in ann["images"]}
        self.instances = {inst["id"]: inst for inst in ann["instances"]}
        self.categories = {cat["id"]: cat["name"] for cat in ann["categories"]}
        self.relationships = {rel["id"]: rel["name"] for rel in ann["relationships"]}

        with open(object_json, 'r') as f:
            object_data = json.load(f)
        object_id2name = {}
        for img_obj in object_data:
            for obj in img_obj.get("objects", []):
                names = obj.get("names", [])
                object_id2name[obj["object_id"]] = names[0] if names else None
        self.object_id2name = object_id2name

        self.annos = [
            anno for anno in ann["annotations"]
            if anno["subject_id"] in self.instances
               and anno["object_id"] in self.instances
               and anno["image_id"] in self.images
        ]

        self.imgid2annos = {}
        for anno in self.annos:
            self.imgid2annos.setdefault(anno["image_id"], []).append(anno)
        self.image_ids = list(self.imgid2annos.keys())

        self.clip_model, self.preprocess = clip.load(clip_model_type, device=device)
        self.cache = FeatureCache(cache_dir) if cache_dir else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        通用写法，既支持多图，也支持单图（如SingleImageVGDataset构造时self.images只含一张图）。
        """
        # 1. 获取当前图片的信息
        # self.images为list或dict，确保都能正确索引
        if isinstance(self.image_ids, list):
            image_id = self.image_ids[idx]
        else:
            # 单张图场景下，允许idx=0，self.image_ids为单元素list或直接为int
            image_id = self.image_ids if isinstance(self.image_ids, int) else list(self.image_ids)[idx]

        annos = self.imgid2annos[image_id]

        # self.images为list或dict都能支持
        # 如果是dict，image_id就是key
        # 如果是list，image_id需要先查找对应下标
        if isinstance(self.images, dict):
            img_info = self.images[image_id]
        elif isinstance(self.images, list):
            # 通用：大多数多图场景下，self.images为list，元素的"id"为image_id
            # 需找到id==image_id的那一项
            img_info = next(img for img in self.images if img.get('id', None) == image_id)
        else:
            raise RuntimeError("self.images must be a list or dict.")

        img_path = os.path.join(self.img_dir_root, img_info["file_name"])
        image_pil = Image.open(img_path).convert("RGB")

        sub_boxes, obj_boxes = [], []
        sub_visual_feats, obj_visual_feats, union_visual_feats = [], [], []
        sub_text_feats, obj_text_feats, phrases, labels = [], [], [], []
        gt_triplets = []

        # 图像全局特征
        if self.cache:
            image_feat = self.cache.load(image_id, "image")
            if image_feat is None:
                image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_feat = self.clip_model.encode_image(image_tensor).squeeze(0).float().cpu()
                self.cache.save(image_feat, image_id, "image")
        else:
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feat = self.clip_model.encode_image(image_tensor).squeeze(0).float().cpu()

        def crop_bbox(image, bbox):
            x, y, w, h = map(int, bbox)
            x2, y2 = x + w, y + h
            x, y = max(0, x), max(0, y)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 <= x or y2 <= y:
                return Image.new("RGB", (1, 1))
            return image.crop((x, y, x2, y2))

        def get_region_feat(region_type, bbox):
            if self.cache:
                feat = self.cache.load(image_id, region_type, bbox=bbox)
                if feat is not None:
                    return feat
            region_img = self.preprocess(crop_bbox(image_pil, bbox)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(region_img).squeeze(0).float().cpu()
            if self.cache:
                self.cache.save(feat, image_id, region_type, bbox=bbox)
            return feat

        def get_text_feat(region_type, text):
            if self.cache:
                feat = self.cache.load(image_id, region_type, text=text)
                if feat is not None:
                    return feat
            with torch.no_grad():
                feat = self.clip_model.encode_text(clip.tokenize([text], truncate=True).to(self.device)).squeeze(
                    0).float().cpu()
            if self.cache:
                self.cache.save(feat, image_id, region_type, text=text)
            return feat

        for anno in annos:
            subject_id = anno["subject_id"]
            object_id = anno["object_id"]
            relation_id = anno["relation_id"]
            category1 = anno["category1"]
            category2 = anno["category2"]

            sub_inst = self.instances[subject_id]
            obj_inst = self.instances[object_id]
            sub_box = torch.tensor(sub_inst["bbox"], dtype=torch.float)
            obj_box = torch.tensor(obj_inst["bbox"], dtype=torch.float)
            sub_boxes.append(sub_box)
            obj_boxes.append(obj_box)

            union_box = [
                min(sub_box[0], obj_box[0]),
                min(sub_box[1], obj_box[1]),
                max(sub_box[0] + sub_box[2], obj_box[0] + obj_box[2]) - min(sub_box[0], obj_box[0]),
                max(sub_box[1] + sub_box[3], obj_box[1] + obj_box[3]) - min(sub_box[1], obj_box[1])
            ]

            sub_visual_feats.append(get_region_feat("sub", sub_box))
            obj_visual_feats.append(get_region_feat("obj", obj_box))
            union_visual_feats.append(get_region_feat("union", union_box))

            sub_text = self.object_id2name.get(subject_id, self.categories.get(category1, "object"))
            if sub_text is None:
                sub_text = self.categories.get(category1, "object")
            obj_text = self.object_id2name.get(object_id, self.categories.get(category2, "object"))
            if obj_text is None:
                obj_text = self.categories.get(category2, "object")
            predicate = self.relationships[relation_id]
            predicate_superclass = predicate2superclass.get(predicate, "unknown")
            phrase = f"{sub_text} {predicate_superclass} {obj_text}"

            sub_text_feats.append(get_text_feat("sub_text", sub_text))
            obj_text_feats.append(get_text_feat("obj_text", obj_text))
            phrases.append(phrase)
            labels.append(relation_id)

            # GT三元组（PredCLS下必须含subject/object类别和box）
            sub_box_xyxy = [
                float(sub_box[0]),
                float(sub_box[1]),
                float(sub_box[0] + sub_box[2]),
                float(sub_box[1] + sub_box[3])
            ]
            obj_box_xyxy = [
                float(obj_box[0]),
                float(obj_box[1]),
                float(obj_box[0] + obj_box[2]),
                float(obj_box[1] + obj_box[3])
            ]
            gt_triplets.append({
                'sub_cls': category1,
                'obj_cls': category2,
                'pred_cls': relation_id,
                'sub_box': sub_box_xyxy,
                'obj_box': obj_box_xyxy
            })

        sub_boxes = torch.stack(sub_boxes)  # (N, 4)
        obj_boxes = torch.stack(obj_boxes)  # (N, 4)
        sub_visual_feats = torch.stack(sub_visual_feats)  # (N, 512)
        obj_visual_feats = torch.stack(obj_visual_feats)  # (N, 512)
        union_visual_feats = torch.stack(union_visual_feats)  # (N, 512)
        sub_text_feats = torch.stack(sub_text_feats)  # (N, 512)
        obj_text_feats = torch.stack(obj_text_feats)  # (N, 512)
        labels = torch.tensor(labels, dtype=torch.long)  # (N,)

        return {
            'image_id': image_id,
            'image': image_feat,  # (512,)
            'sub_boxes': sub_boxes,  # (N,4)
            'obj_boxes': obj_boxes,  # (N,4)
            'sub_visual_feats': sub_visual_feats,  # (N,512)
            'obj_visual_feats': obj_visual_feats,  # (N,512)
            'union_visual_feats': union_visual_feats,  # (N,512)
            'sub_text_feats': sub_text_feats,  # (N,512)
            'obj_text_feats': obj_text_feats,  # (N,512)
            'phrases': phrases,  # list[str]
            'labels': labels,  # (N,)
            'gt_triplets': gt_triplets  # List[dict]
        }

def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [b[key] for b in batch if key in b]
        # 只对长度一致的tensor stack，否则用list
        if isinstance(vals[0], torch.Tensor) and all(v.shape == vals[0].shape for v in vals):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out