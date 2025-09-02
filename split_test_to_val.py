# split_test_to_val.py
import json
import random
import os

def split_dataset(input_json_path, val_ratio=0.2, random_seed=42):
    """
    将输入的场景图JSON文件按图像ID划分为新的测试集和验证集。

    参数:
    input_json_path (str): 原始测试集JSON文件的路径。
    val_ratio (float): 划分给验证集的图像比例 (0.0 到 1.0 之间)。
    random_seed (int): 随机种子，用于可复现的划分。
    """
    print(f"开始处理文件: {input_json_path}")
    print(f"验证集划分比例: {val_ratio*100:.2f}%")

    try:
        with open(input_json_path, 'r') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_json_path}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{input_json_path}' 不是有效的JSON格式。")
        return
    except Exception as e:
        print(f"加载文件 '{input_json_path}' 时发生错误: {e}")
        return

    # 设置随机种子
    random.seed(random_seed)

    # 1. 获取所有图像信息并打乱
    all_images_info = original_data['images']
    random.shuffle(all_images_info) # 直接打乱列表

    # 2. 划分图像信息和图像ID
    num_total_images = len(all_images_info)
    num_val_images = int(num_total_images * val_ratio)
    num_new_test_images = num_total_images - num_val_images

    val_images_info = all_images_info[:num_val_images]
    new_test_images_info = all_images_info[num_val_images:]

    val_image_ids = {img['id'] for img in val_images_info}
    new_test_image_ids = {img['id'] for img in new_test_images_info}

    print(f"原始图像总数: {num_total_images}")
    print(f"划分给验证集的图像数: {len(val_image_ids)}")
    print(f"划分给新测试集的图像数: {len(new_test_image_ids)}")

    # 3. 筛选 annotations, instances
    original_annotations = original_data['annotations']
    original_instances = original_data['instances']

    val_annotations = [ann for ann in original_annotations if ann['image_id'] in val_image_ids]
    new_test_annotations = [ann for ann in original_annotations if ann['image_id'] in new_test_image_ids]

    val_instances = [inst for inst in original_instances if inst['image_id'] in val_image_ids]
    new_test_instances = [inst for inst in original_instances if inst['image_id'] in new_test_image_ids]

    print(f"验证集关系标注数: {len(val_annotations)}")
    print(f"新测试集关系标注数: {len(new_test_annotations)}")
    print(f"验证集物体实例数: {len(val_instances)}")
    print(f"新测试集物体实例数: {len(new_test_instances)}")

    # 4. 准备新的JSON数据结构
    val_dataset = {
        'images': val_images_info,
        'annotations': val_annotations,
        'categories': original_data['categories'], # 类别定义是共享的
        'instances': val_instances,
        'relationships': original_data['relationships'] # 关系定义是共享的
    }

    new_test_dataset = {
        'images': new_test_images_info,
        'annotations': new_test_annotations,
        'categories': original_data['categories'],
        'instances': new_test_instances,
        'relationships': original_data['relationships']
    }

    # 5. 保存到新文件
    base_dir = os.path.dirname(input_json_path)
    original_filename = os.path.basename(input_json_path)
    name_part, ext_part = os.path.splitext(original_filename)

    val_output_path = os.path.join(base_dir, f"{name_part}_val.json")
    new_test_output_path = os.path.join(base_dir, f"{name_part}_new_test.json")

    try:
        with open(val_output_path, 'w') as f_val:
            json.dump(val_dataset, f_val, indent=2) # indent可选，用于美化输出
        print(f"验证集已保存到: {val_output_path}")

        with open(new_test_output_path, 'w') as f_new_test:
            json.dump(new_test_dataset, f_new_test, indent=2)
        print(f"新测试集已保存到: {new_test_output_path}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == "__main__":
    # --------------------------------------------------------------------
    # TODO: 用户需要在此处设置原始测试集JSON文件路径!
    # --------------------------------------------------------------------
    original_test_json_path = "dataset/vg/annotations/instances_vg_test.json" # 修改为你的路径

    # 可以选择验证集比例和随机种子
    validation_ratio = 0.6  # 例如，从测试集中取50%作为验证集 (根据你的需求调整)
                            # VG测试集通常比较大，可以多分一些给验证集
                            # 常见的做法可能是从原始训练集中分验证集，测试集保持不动用于最终评估
                            # 但既然你的需求是从测试集分，我们就这么做。
    seed_for_splitting = 123 # 保证每次划分结果一致

    if not os.path.exists(original_test_json_path):
        print(f"错误: 输入文件 '{original_test_json_path}' 未找到。请检查路径。")
    else:
        split_dataset(original_test_json_path,
                      val_ratio=validation_ratio,
                      random_seed=seed_for_splitting)