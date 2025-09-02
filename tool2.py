import h5py
import numpy as np


def explore_h5_file(file_path):
    """
    递归探索 HDF5 文件的结构并打印详细信息。
    :param file_path: HDF5 文件的路径
    """

    def explore_group(group, indent=0):
        """
        递归探索 HDF5 组的结构。
        :param group: 当前的 HDF5 组
        :param indent: 当前的缩进级别
        """
        for key in group.keys():
            item = group[key]
            print("  " * indent + f"- {key}:", end=" ")
            if isinstance(item, h5py.Group):
                print("Group")
                explore_group(item, indent + 1)
            elif isinstance(item, h5py.Dataset):
                print(f"Dataset, shape: {item.shape}, dtype: {item.dtype}")
                # 打印数据集的前几行数据
                print("  " * (indent + 1) + "Sample data:")
                sample_data = item[:5]  # 获取前5个数据点
                print("  " * (indent + 1) + f"{sample_data}")
                # 打印数据集的统计信息
                if item.dtype.kind in 'bifc':  # 检查是否为数值类型
                    print("  " * (indent + 1) + "Statistics:")
                    print("  " * (indent + 1) + f"  Min: {np.min(item)}")
                    print("  " * (indent + 1) + f"  Max: {np.max(item)}")
                    print("  " * (indent + 1) + f"  Mean: {np.mean(item)}")
                    print("  " * (indent + 1) + f"  Std: {np.std(item)}")
            else:
                print(f"Unknown type: {type(item)}")

    with h5py.File(file_path, 'r') as file:
        print(f"Exploring HDF5 file: {file_path}")
        explore_group(file)


if __name__ == "__main__":

    file_path = "dataset/vg/annotations/VG-SGG-with-attri.h5"
    explore_h5_file(file_path)