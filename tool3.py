import h5py


def explore_hdf5_keys(file_path):
    try:
        with h5py.File(file_path, 'r') as hf:
            print(f"Keys in HDF5 file: {file_path}")
            for key in hf.keys():
                print(f"- {key}: Type: {type(hf[key])}")
                if isinstance(hf[key], h5py.Dataset):
                    print(f"  Shape: {hf[key].shape}, Dtype: {hf[key].dtype}")
                    # Optionally print a small sample if it's an ID-like field
                    if 'id' in key.lower() and hf[key].ndim == 1 and hf[key].shape[0] > 0:
                        print(f"  Sample IDs: {hf[key][:5]}")

    except Exception as e:
        print(f"Error reading HDF5 file: {e}")


if __name__ == '__main__':
    hdf5_file = 'dataset/vg/annotations/VG-SGG-with-attri.h5'  # Your path
    explore_hdf5_keys(hdf5_file)
