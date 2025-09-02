import json


def debug_triplet_loss(ann_file, image_id):
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 构建instances id集合
    instance_ids = set(inst['id'] for inst in data['instances'])

    # 统计该图片的全部GT关系三元组
    gt_annos = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    print(f"图片 {image_id} 的原始GT三元组数量: {len(gt_annos)}")

    # 检查每个三元组的subject_id/object_id是否都在实例中
    kept = []
    for ann in gt_annos:
        sid, oid = ann['subject_id'], ann['object_id']
        missing = []
        if sid not in instance_ids:
            missing.append(f"subject_id缺失: {sid}")
        if oid not in instance_ids:
            missing.append(f"object_id缺失: {oid}")
        if missing:
            print(f"被过滤三元组: {ann}，原因: {', '.join(missing)}")
        else:
            print(f"正常保留三元组: {ann}")
            kept.append(ann)
    print(f"通过代码实际保留的三元组数量: {len(kept)}")


if __name__ == "__main__":
    ann_file = "dataset/vg/annotations/instances_vg_test_new_test.json"
    image_id = 2343720  # 你要检查的图片id
    debug_triplet_loss(ann_file, image_id)
