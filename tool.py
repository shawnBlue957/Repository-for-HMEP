# tool.py
import json
import os
from collections.abc import Iterable


def print_json_structure(data, indent=0, max_list_items_to_show=2, max_dict_items_to_show=10):
    """
    递归打印JSON数据的结构，包括类型和示例。
    """
    prefix = "  " * indent
    if isinstance(data, dict):
        for i, (key, value) in enumerate(data.items()):
            if i >= max_dict_items_to_show and max_dict_items_to_show != -1:
                print(f"{prefix}  ... (and {len(data) - max_dict_items_to_show} more keys)")
                break
            print(f"{prefix}  Key: '{key}'")
            print_json_structure(value, indent + 2, max_list_items_to_show, max_dict_items_to_show)
    elif isinstance(data, list):
        print(f"{prefix}List (length: {len(data)})")
        if len(data) > 0:
            items_to_show = min(len(data), max_list_items_to_show)
            for i in range(items_to_show):
                print(f"{prefix}  Item {i}:")
                print_json_structure(data[i], indent + 2, max_list_items_to_show, max_dict_items_to_show)
            if len(data) > max_list_items_to_show and max_list_items_to_show != -1:
                print(f"{prefix}  ... (and {len(data) - max_list_items_to_show} more items)")
        else:
            print(f"{prefix}  (empty list)")
    else:
        print(f"{prefix}Value: {data} (Type: {type(data).__name__})")


def inspect_scene_graph_json(file_path):
    """
    加载并检查场景图JSON文件的结构。
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        return
    except Exception as e:
        print(f"加载文件 '{file_path}' 时发生错误: {e}")
        return

    print(f"--- 开始检查JSON文件: {file_path} ---")
    print("\n顶层键及其基本信息:")
    top_level_keys = list(data.keys())
    print(f"  {top_level_keys}")

    common_sg_keys = ['images', 'annotations', 'categories', 'instances', 'relationships']

    for key in top_level_keys:
        print(f"\n--- 详细结构: '{key}' ---")
        value = data[key]
        if isinstance(value, list):
            print(f"  '{key}' 是一个列表，包含 {len(value)} 个元素。")
            if len(value) > 0:
                print(f"  '{key}' 列表的第一个元素结构:")
                # 对于常见的场景图键，我们可能想看更详细的结构
                if key in common_sg_keys:
                    print_json_structure(value[0], indent=2, max_list_items_to_show=2,
                                         max_dict_items_to_show=-1)  # 显示字典所有键
                else:
                    print_json_structure(value[0], indent=2)

                # 如果列表元素类型可能不同，可以考虑多显示几个
                if len(value) > 1:
                    print(f"\n  '{key}' 列表的第二个元素结构 (如果存在且不同):")
                    try:
                        # 简单比较类型，如果第一个是字典，第二个也是字典，则认为结构相似
                        if type(value[0]) == type(value[1]) and isinstance(value[0], dict):
                            # 可以加入更复杂的结构比较逻辑
                            if value[0].keys() != value[1].keys():
                                print("  (第二个元素与第一个元素键不同)")
                                if key in common_sg_keys:
                                    print_json_structure(value[1], indent=2, max_list_items_to_show=2,
                                                         max_dict_items_to_show=-1)
                                else:
                                    print_json_structure(value[1], indent=2)
                            else:
                                print("  (第二个元素与第一个元素键相同，结构类似，不再赘述)")
                        elif type(value[0]) != type(value[1]):
                            print_json_structure(value[1], indent=2)

                    except IndexError:
                        pass  # 列表只有一个元素
            else:
                print(f"  '{key}' 是一个空列表。")

        elif isinstance(value, dict):
            print(f"  '{key}' 是一个字典。")
            print(f"  '{key}' 字典的键:")
            print_json_structure(value, indent=2, max_list_items_to_show=1, max_dict_items_to_show=-1)  # 显示字典所有键
        else:
            print(f"  '{key}' 的值是: {value} (类型: {type(value).__name__})")

    print(f"\n--- JSON文件检查完毕: {file_path} ---")


if __name__ == "__main__":
    # --------------------------------------------------------------------
    json_file_to_inspect = "dataset/vg/annotations/instances_vg_train.json"
    # 或者: json_file_to_inspect = "dataset/vg/annotations/instances_vg_train.json"

    if not json_file_to_inspect:
        print("未输入文件路径，将使用默认示例路径（如果存在）。")
        # 你可以在这里设置一个默认路径，或者直接退出
        # default_path = "path/to/your/default.json"
        # if os.path.exists(default_path):
        #    json_file_to_inspect = default_path
        # else:
        #    print("默认路径不存在，退出。")
        #    exit()
        json_file_to_inspect = "instances_vg_test.json"  # 假设当前目录有此文件用于测试
        print(f"尝试使用默认路径: {json_file_to_inspect}")

    if os.path.exists(json_file_to_inspect):
        inspect_scene_graph_json(json_file_to_inspect)
    else:
        print(f"错误: 文件 '{json_file_to_inspect}' 未找到。请确保路径正确。")
