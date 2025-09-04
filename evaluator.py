import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_iou(box1, box2):
    """Calculate the IoU of two bboxes，box: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def match_triplet(pred, gt, iou_thr=0.5):
    """Determine whether a single predicted triple matches the GT triple"""
    return (
        pred['sub_cls'] == gt['sub_cls'] and
        pred['obj_cls'] == gt['obj_cls'] and
        pred['pred_cls'] == gt['pred_cls']
    )

def evaluate_sgg_recall_by_image(
        model, dataloader, recall_ks, device="cuda", iou_thr=0.5
):
    """
    Evaluate Recall@k on an image-by-image basis (scene graph/PredCLS criterion)
    - model: must implement model.predict_triplets(batch)
    - dataloader: Each batch returns a dict, each key is a list, and each image contains a GT triplet
    """
    model.eval()
    total_gt = 0
    recall_hits = {k: 0 for k in recall_ks}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating SGG Recall@k(image-level)")):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            gt_triplets_list = batch['gt_triplets']
            pred_triplets_list = model.predict_triplets(batch, device=device)
            for img_idx, (gt_list, pred_list) in enumerate(zip(gt_triplets_list, pred_triplets_list)):
                total_gt += len(gt_list)
                pred_list_sorted = sorted(pred_list, key=lambda x: x['score'], reverse=True)
                for k in recall_ks:
                    topk_pred = pred_list_sorted[:k]
                    matched_gt = set()
                    for gt_id, gt in enumerate(gt_list):
                        for pred in topk_pred:
                            if match_triplet(pred, gt, iou_thr):
                                matched_gt.add(gt_id)
                                break
                    recall_hits[k] += len(matched_gt)
    for k in recall_ks:
        recall = recall_hits[k] / max(1, total_gt)
        print(f"[SceneGraph][PredCLS] Recall@{k}: {recall:.4f}")

if __name__ == "__main__":
    # ===== Configure parameters directly here =====
    ann_file = "dataset/vg/annotations/instances_vg_test_new_test.json"
    img_dir_root = "dataset/vg/images"
    object_json = "dataset/vg/annotations/objects.json"
    clip_model_type = "ViT-B/32"
    cache_dir = "./feature_cache"
    batch_size = 32
    ckpt = "checkpoints_multihead_attention_v2/vg_relation_best.pth"
    recall_ks = [1, 5, 10, 15, 20, 50, 100]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # iou_thr = 0.5
    # ========== The structural parameters need to be exactly the same as during training ==========
    fusion_type = "multihead_attention"
    fusion_heads = 4
    fusion_depth = 2
    fused_feature_dim = 256
    dropout_p = 0.3
    use_visual = True
    use_text = True
    use_coord = True
    # =================================

    from dataloader import VGRelationDataset, collate_fn
    from model import VGRelationModel

    dataset = VGRelationDataset(
        ann_file=ann_file,
        img_dir_root=img_dir_root,
        object_json=object_json,
        clip_model_type=clip_model_type,
        device=device,
        cache_dir=cache_dir
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = VGRelationModel(
        num_predicates=len(dataset.relationships),
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

    evaluate_sgg_recall_by_image(model, dataloader, recall_ks, device=device)

    # , iou_thr=iou_thr
