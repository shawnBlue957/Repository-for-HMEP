import torch
import json
import os
from model import VGRelationModel
from dataloader import VGRelationDataset, collate_fn

def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def get_image_info_by_id(ann_file, target_img_id):
    with open(ann_file, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    images = ann['images']
    for img in images:
        if img['id'] == target_img_id:
            return img['id'], img['file_name']
    raise ValueError(f"image_id={target_img_id} Not found in annotation file！")

class SingleImageVGDataset(VGRelationDataset):
    def __init__(self, img_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.images, dict):
            images_list = list(self.images.values())
        else:
            images_list = self.images
        if not images_list:
            raise ValueError("self.images Empty, please check the dataset initialization.")
        if isinstance(images_list[0], dict):
            self.images = [img for img in images_list if img['id'] == img_id]
        else:
            self.images = [img_id]
        if hasattr(self, 'annotations') and self.annotations is not None:
            self.annotations = [ann for ann in self.annotations if ann['image_id'] == img_id]
        self.image_ids = [img_id]
        self.ids = [img_id]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return super().__getitem__(0)

def debug_triplet_loss(ann_file, image_id, id2name, relid2name):
    print(f"\n========GT triplet detailed inspection（debug_triplet_loss） ========")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build instance id collection
    instance_ids = set(inst['id'] for inst in data['instances'])

    # Count all GT relationship triples of the image
    gt_annos = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    print(f"image {image_id} The number of original GT triplets: {len(gt_annos)}")

    # Check if the subject_id/object_id of each triple is in the instance
    kept = []
    for ann in gt_annos:
        sid, oid = ann['subject_id'], ann['object_id']
        missing = []
        if sid not in instance_ids:
            missing.append(f"subject_id Missing: {sid}")
        if oid not in instance_ids:
            missing.append(f"object_id Missing: {oid}")
        sub_cls_name = id2name.get(ann['category1'], f"unknow({ann['category1']})")
        obj_cls_name = id2name.get(ann['category2'], f"unknow({ann['category2']})")
        rel_cls_name = relid2name.get(ann['relation_id'], f"unknow({ann['relation_id']})")
        if missing:
            print(f"Filtered triples: Subject: {sub_cls_name}({ann['category1']}) -[{rel_cls_name}({ann['relation_id']})]-> object: {obj_cls_name}({ann['category2']})，reason: {', '.join(missing)}")
        else:
            print(f"Normally retain triples: subject: {sub_cls_name}({ann['category1']}) -[{rel_cls_name}({ann['relation_id']})]-> object: {obj_cls_name}({ann['category2']})")
            kept.append(ann)
    print(f"The number of triplets actually retained by the code: {len(kept)}")
    print("====================================================\n")
    return kept

def build_id2name_dicts(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = data['categories']
    relationships = data['relationships']
    id2name = {cat['id']: cat['name'] for cat in categories}
    relid2name = {rel['id']: rel['name'] for rel in relationships}
    return id2name, relid2name

if __name__ == "__main__":
    # ===== Configuration parameters =====
    ann_file = "dataset/vg/annotations/instances_vg_test_new_test.json"
    img_dir_root = "dataset/vg/images"
    object_json = "dataset/vg/annotations/objects.json"
    clip_model_type = "ViT-B/32"
    cache_dir = "./feature_cache"
    ckpt = "checkpoints_gate_v1/vg_relation_best.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_type = "gate"
    fusion_heads = 4
    fusion_depth = 2
    fused_feature_dim = 256
    dropout_p = 0.3
    use_visual = True
    use_text = True
    use_coord = True
    target_img_id = 2343720  # The specified image id

    # ====== Build category and relationship id->name mapping ======
    id2name, relid2name = build_id2name_dicts(ann_file)

    # ======= 1. Print GT triple retention and filtering information (with category name) =======
    kept_gt_triplets = debug_triplet_loss(ann_file, target_img_id, id2name, relid2name)

    # ======= 2. Predict and print the TOP50 score triplets, with the category and relation name. =======
    img_id, img_file = get_image_info_by_id(ann_file, target_img_id)
    print(f"\nSpecify an image: {img_file} (image_id={img_id})")

    dataset = SingleImageVGDataset(
        img_id,
        ann_file=ann_file,
        img_dir_root=img_dir_root,
        object_json=object_json,
        clip_model_type=clip_model_type,
        device=device,
        cache_dir=cache_dir
    )
    print("SingleImageVGDatasetlength：", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    num_predicates = len(dataset.relationships)

    model = VGRelationModel(
        num_predicates=num_predicates,
        clip_model_type=clip_model_type,
        fused_feature_dim=fused_feature_dim,
        fusion_type=fusion_type,
        fusion_heads=fusion_heads,
        fusion_depth=fusion_depth,
        dropout_p=dropout_p,
        use_visual=use_visual,
        use_text=use_text,
        use_coord=use_coord
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        got_data = False
        for batch in dataloader:
            got_data = True
            batch = to_device(batch, device)
            pred_triplets_list = model.predict_triplets(batch, device=device)
            if not pred_triplets_list or len(pred_triplets_list) == 0:
                print("No triples detected")
                continue
            pred_triplets = pred_triplets_list[0]

            # ------ Sort by score in descending order, only keep the TOP50 ------
            pred_triplets_sorted = sorted(pred_triplets, key=lambda x: x['score'], reverse=True)
            topk = min(50, len(pred_triplets_sorted))
            pred_triplets_sorted = pred_triplets_sorted[:topk]

            print(f"\nThe triplet of the predicted score TOP{topk} is as follows：")
            for idx, triplet in enumerate(pred_triplets_sorted, 1):
                sub_cls_id = triplet['sub_cls']
                obj_cls_id = triplet['obj_cls']
                pred_cls_id = triplet['pred_cls']
                sub_cls_name = id2name.get(sub_cls_id, f"unknow({sub_cls_id})")
                obj_cls_name = id2name.get(obj_cls_id, f"unknow({obj_cls_id})")
                pred_cls_name = relid2name.get(pred_cls_id, f"unknow({pred_cls_id})")
                print(f"TOP{idx}: sub: {sub_cls_name}({sub_cls_id})  obj: {obj_cls_name}({obj_cls_id})  rel: {pred_cls_name}({pred_cls_id})")
                print(f"      sub box: {triplet['sub_box']}  obj box: {triplet['obj_box']}  score: {triplet['score']:.4f}\n")
        if not got_data:

            print("Warning: Dataloader did not generate any batches! Please check your dataset construction.")

