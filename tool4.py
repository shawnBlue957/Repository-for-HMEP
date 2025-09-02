import json
import os


def load_and_inspect_vg_annotations(file_path):
    """
    加载并检查 Visual Genome 数据集的注释文件。

    参数:
        file_path (str): 注释文件的路径。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在。")
        return

    # 加载 JSON 文件
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 打印数据的基本信息
    print("Visual Genome 数据集注释文件分析：")
    print(f"文件路径: {file_path}")
    print(f"数据类型: {type(data)}")
    print(f"数据长度: {len(data)}")

    # 检查数据的结构
    print("\n字典的键：")
    print(list(data.keys()))

    # 检查 vg_coco_id_to_capgraphs 键的内容
    if 'vg_coco_id_to_capgraphs' in data:
        print("\n'vg_coco_id_to_capgraphs' 键的内容：")
        capgraphs = data['vg_coco_id_to_capgraphs']
        print(f"包含的 COCO 图像 ID 数量: {len(capgraphs)}")

        # 打印第一个 COCO 图像 ID 的 capgraph 内容
        first_coco_id = list(capgraphs.keys())[0]
        print(f"\n第一个 COCO 图像 ID ({first_coco_id}) 的 capgraph 内容：")
        print(json.dumps(capgraphs[first_coco_id], indent=4))

        # 统计关系类别
        predicate_counts = {}
        for capgraph in capgraphs.values():
            for cap in capgraph:
                for relation in cap.get('relations', []):
                    predicate = relation[2]
                    if predicate in predicate_counts:
                        predicate_counts[predicate] += 1
                    else:
                        predicate_counts[predicate] = 1

        # 打印关系类别的统计信息
        if predicate_counts:
            print("\n关系类别的统计信息：")
            for predicate, count in sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"{predicate}: {count}")
            print(f"\n总关系类别数: {len(predicate_counts)}")
        else:
            print("\n未找到关系类别统计信息。")
    else:
        print("\n未找到 'vg_coco_id_to_capgraphs' 键。")


if __name__ == "__main__":
    # 设置注释文件的路径
    file_path = "dataset/vg/annotations/vg_capgraphs_anno.json"
    load_and_inspect_vg_annotations(file_path)