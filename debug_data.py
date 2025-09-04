import json


def debug_triplet_loss(ann_file, image_id):
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build instance id collection
    instance_ids = set(inst['id'] for inst in data['instances'])

    # Count all GT relationship triples of the image
    gt_annos = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    print(f"The number of original GT triplets of image {image_id}:{len(gt_annos)}")

    # Check if the subject_id/object_id of each triple is in the instance
    kept = []
    for ann in gt_annos:
        sid, oid = ann['subject_id'], ann['object_id']
        missing = []
        if sid not in instance_ids:
            missing.append(f"subject_id Missing: {sid}")
        if oid not in instance_ids:
            missing.append(f"object_id Missing: {oid}")
        if missing:
            print(f"Filtered triples: {ann}, reason: {', '.join(missing)}")
        else:
            print(f"Normal retention of triples: {ann}")
            kept.append(ann)
    print(f"The number of triplets actually retained by the code: {len(kept)}")


if __name__ == "__main__":
    ann_file = "dataset/vg/annotations/instances_vg_test_new_test.json"
    image_id = 2343720  # The id of the image you want to check
    debug_triplet_loss(ann_file, image_id)

