# tool.py
import torch
import os
import sys
import numpy as np

def view_pkl_file(file_path):
    """
    Loads a .pkl file (saved with torch.save) and prints its contents.
    """
    if not os.path.exists(file_path):
        print(f"错误：文件未找到于 '{file_path}'")
        sys.exit(1)

    print(f"正在加载文件：{file_path}")
    try:
        data = torch.load(file_path)
        print("\n--- 文件内容概述 ---")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\n键: '{key}'")
                print(f"  类型: {type(value)}")

                if isinstance(value, torch.Tensor):
                    print(f"  形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    # 对于大型Tensor，只打印部分内容
                    if value.numel() > 100:
                        print(f"  值 (前5个元素): {value.flatten()[:5]}...")
                    else:
                        print(f"  值: {value}")
                elif isinstance(value, list):
                    print(f"  元素数量: {len(value)}")
                    if len(value) > 0:
                        first_item = value[0]
                        print(f"  第一个元素类型: {type(first_item)}")
                        if isinstance(first_item, torch.Tensor):
                            print(f"  第一个元素形状: {first_item.shape}")
                            print(f"  第一个元素数据类型: {first_item.dtype}")
                            if first_item.numel() > 50:
                                print(f"  第一个元素值 (部分): {first_item.flatten()[:5]}...")
                            else:
                                print(f"  第一个元素值: {first_item}")
                        else:
                            # 对于非Tensor列表，打印前几个元素
                            if len(value) <= 5:
                                print(f"  值 (全部): {value}")
                            else:
                                print(f"  值 (前5个元素): {value[:5]}...")
                elif isinstance(value, np.ndarray):
                    print(f"  形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    if value.size > 100:
                        print(f"  值 (前5个元素): {value.flatten()[:5]}...")
                    else:
                        print(f"  值: {value}")
                else:
                    print(f"  值: {value}")
        else:
            print(f"加载的内容不是字典。类型: {type(data)}")
            # 如果不是字典，直接打印数据（小心大型对象）
            if isinstance(data, torch.Tensor):
                print(f"  形状: {data.shape}")
                print(f"  数据类型: {data.dtype}")
                if data.numel() > 100:
                    print(f"  值 (前5个元素): {data.flatten()[:5]}...")
                else:
                    print(f"  值: {data}")
            else:
                print(f"  值: {data}")

        print("\n--- 文件内容概述结束 ---")

    except Exception as e:
        print(f"加载或解析文件时发生错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    # 假设你的项目根目录是 "your_project_root/"
    # 并且 dataset/vg_scene_graph_annot/VG_100K/2_annotations.pkl 文件存在于此
    file_to_view = 'dataset/vg_scene_graph_annot/VG_100K/2_annotations.pkl'
    view_pkl_file(file_to_view)
