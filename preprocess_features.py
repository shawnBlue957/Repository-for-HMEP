# preprocess_features.py
import os
from datetime import datetime

import h5py
import torch
import clip
import numpy as np
from tqdm import tqdm
import json  # For config

# Import VGDataset and other necessary components from your dataloader
# Assuming dataloader.py is in the same directory or accessible via PYTHONPATH
from dataloader import VGDataset

# --- Configuration ---
# These should match the paths used by your VGDataset and point to your raw data
# It's good practice to load these from a config file or command-line arguments
# For this example, I'll define them here.
# Ensure these paths are correct for your system!
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # Assumes script is in project root with 'dataset' subdir
CONFIG = {
    "hdf5_path": os.path.join(BASE_DATA_DIR, 'dataset/vg/annotations/VG-SGG-with-attri.h5'),
    "image_dir": os.path.join(BASE_DATA_DIR, 'dataset/vg/images/'),
    "dict_json_path": os.path.join(BASE_DATA_DIR, 'dataset/vg/annotations/VG-SGG-dicts.json'),
    "capgraphs_anno_path": os.path.join(BASE_DATA_DIR, 'dataset/vg/annotations/vg_capgraphs_anno.json'),
    "output_hdf5_path": os.path.join(BASE_DATA_DIR, 'dataset/vg/annotations/precomputed_clip_features.h5'),
    "clip_model_type": "ViT-B/32",
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available for faster preprocessing
    "verbose_dataset_warnings": False,  # For VGDataset internal warnings
    "batch_size_preprocessing": 256,  # Batching CLIP image encoding if possible (more advanced)
    # Current VGDataset __getitem__ is sample-wise.
    # For true batching, VGDataset would need modification or a different approach.
    # Here, tqdm will iterate sample by sample.
}


# --- Main Preprocessing Function ---
def preprocess_and_save_features(config):
    print(f"Starting feature preprocessing with device: {config['device']}")
    print(f"Output will be saved to: {config['output_hdf5_path']}")

    # Initialize VGDataset to load all data for preprocessing
    # Note: The 'device' param for VGDataset is for its internal CLIP model.
    dataset = VGDataset(
        hdf5_path=config["hdf5_path"],
        image_dir=config["image_dir"],
        vg_dict_json_path=config["dict_json_path"],
        vg_capgraphs_anno_path=config["capgraphs_anno_path"],
        split='all_for_preprocessing',  # Special split to load all relationships
        clip_model_type=config["clip_model_type"],
        device=config["device"],  # Use the main device for CLIP ops in dataset for preprocessing
        verbose_warnings=config["verbose_dataset_warnings"]
        # train_ratio, val_ratio, seed are not needed for 'all_for_preprocessing'
    )

    if len(dataset) == 0:
        print("No data found in dataset for preprocessing. Exiting.")
        return

    print(f"Total relationships to preprocess: {len(dataset)}")

    # Prepare HDF5 file for output
    # We will store features for all relationships. Splitting will be done by the new Dataloader.
    with h5py.File(config["output_hdf5_path"], 'w') as out_hf:
        # Determine feature sizes (e.g., CLIP ViT-B/32 is 512)
        # Get one sample to infer feature dimensions
        try:
            print("Fetching a sample item to determine feature dimensions...")
            sample_item = dataset[0]  # Get the first full item
            if sample_item is None:  # Should not happen if get_dummy_item returns valid structure
                raise ValueError("First sample item is None, cannot determine feature dimensions.")
        except Exception as e:
            print(f"Error getting sample item: {e}. Assuming CLIP ViT-B/32 (512 dim).")
            # If dataset[0] fails (e.g. first image is bad and returns dummy with specific error)
            # We still need feature dimensions. Let's try to load CLIP model to get it.
            try:
                temp_clip_model, _ = clip.load(config["clip_model_type"], device="cpu")  # Load on CPU to be safe
                clip_feat_dim = temp_clip_model.visual.output_dim
                del temp_clip_model  # Free memory
            except Exception:
                clip_feat_dim = 512  # Fallback
            print(f"Using feature dimension: {clip_feat_dim}")
            # Create dummy sample structure if first sample failed
            dummy_feat = np.zeros(clip_feat_dim, dtype=np.float32)
            sample_item = {
                'image': dummy_feat, 'sub_text': dummy_feat, 'obj_text': dummy_feat,
                'sub_box': np.zeros(4, dtype=np.float32), 'obj_box': np.zeros(4, dtype=np.float32),
                'sub_visual_feat': dummy_feat, 'obj_visual_feat': dummy_feat,
                'union_visual_feat': dummy_feat, 'label': np.array([0], dtype=np.int64),
                'phrase': "error_sample_phrase"
            }

        img_feat_dim = sample_item['image'].shape[0]  # Global image feature
        text_feat_dim = sample_item['sub_text'].shape[0]  # Subject text feature
        box_dim = sample_item['sub_box'].shape[0]  # Box [x,y,w,h]
        vis_feat_dim = sample_item['sub_visual_feat'].shape[0]  # Region visual feature

        print(f"  Image feature dimension: {img_feat_dim}")
        print(f"  Text feature dimension: {text_feat_dim}")
        print(f"  Box dimension: {box_dim}")
        print(f"  Region visual feature dimension: {vis_feat_dim}")

        # Create datasets in HDF5
        num_samples = len(dataset)
        dset_global_img = out_hf.create_dataset('global_image_features', (num_samples, img_feat_dim), dtype='f4')
        dset_sub_text = out_hf.create_dataset('subject_text_features', (num_samples, text_feat_dim), dtype='f4')
        dset_obj_text = out_hf.create_dataset('object_text_features', (num_samples, text_feat_dim), dtype='f4')
        dset_sub_vis = out_hf.create_dataset('subject_visual_features', (num_samples, vis_feat_dim), dtype='f4')
        dset_obj_vis = out_hf.create_dataset('object_visual_features', (num_samples, vis_feat_dim), dtype='f4')
        dset_union_vis = out_hf.create_dataset('union_visual_features', (num_samples, vis_feat_dim), dtype='f4')
        dset_sub_box = out_hf.create_dataset('subject_boxes', (num_samples, box_dim), dtype='f4')
        dset_obj_box = out_hf.create_dataset('object_boxes', (num_samples, box_dim), dtype='f4')
        dset_labels = out_hf.create_dataset('predicate_labels', (num_samples,), dtype='i8')  # single int label

        # For mapping back to original HDF5 and for splitting later
        dset_hdf5_img_indices = out_hf.create_dataset('original_hdf5_img_indices', (num_samples,), dtype='i4')
        dset_global_rel_indices = out_hf.create_dataset('original_global_rel_indices', (num_samples,), dtype='i4')

        # Optional: store phrases if needed (variable length strings are tricky in HDF5 fixed arrays)
        # For phrases, it's often better to store them in a companion JSON or handle text separately.
        # If you must store in HDF5:
        # dt = h5py.string_dtype(encoding='utf-8')
        # dset_phrases = out_hf.create_dataset('phrases', (num_samples,), dtype=dt)

        print("Iterating through dataset to extract and save features...")
        for i in tqdm(range(num_samples), desc="Preprocessing features"):
            try:
                item = dataset[i]  # This calls VGDataset.__getitem__
                if item is None:  # Should be handled by get_dummy_item now
                    print(f"Warning: Skipping None item at index {i}")
                    # Fill with zeros or a special marker if necessary, or skip.
                    # For now, if get_dummy_item works, this path might not be hit.
                    # If it is hit, it means get_dummy_item also failed or returned None.
                    # To maintain array length, we must write *something*.
                    # Let's assume get_dummy_item gives a valid structure.
                    item = dataset.get_dummy_item(f"Original item at index {i} was None or unrecoverable")

                dset_global_img[i] = item['image'].numpy() if isinstance(item['image'], torch.Tensor) else item['image']
                dset_sub_text[i] = item['sub_text'].numpy() if isinstance(item['sub_text'], torch.Tensor) else item[
                    'sub_text']
                dset_obj_text[i] = item['obj_text'].numpy() if isinstance(item['obj_text'], torch.Tensor) else item[
                    'obj_text']
                dset_sub_vis[i] = item['sub_visual_feat'].numpy() if isinstance(item['sub_visual_feat'],
                                                                                torch.Tensor) else item[
                    'sub_visual_feat']
                dset_obj_vis[i] = item['obj_visual_feat'].numpy() if isinstance(item['obj_visual_feat'],
                                                                                torch.Tensor) else item[
                    'obj_visual_feat']
                dset_union_vis[i] = item['union_visual_feat'].numpy() if isinstance(item['union_visual_feat'],
                                                                                    torch.Tensor) else item[
                    'union_visual_feat']
                dset_sub_box[i] = item['sub_box'].numpy() if isinstance(item['sub_box'], torch.Tensor) else item[
                    'sub_box']
                dset_obj_box[i] = item['obj_box'].numpy() if isinstance(item['obj_box'], torch.Tensor) else item[
                    'obj_box']
                dset_labels[i] = item['label'].item() if isinstance(item['label'], torch.Tensor) else item[
                    'label']  # Ensure scalar

                # Store original HDF5 image index and global relationship index
                # This requires rel_to_img_idx_map to be populated correctly for 'all_for_preprocessing'
                # And filtered_rel_indices to be the global relationship indices
                original_global_rel_idx = dataset.filtered_rel_indices[i]
                original_hdf5_img_idx = dataset.rel_to_img_idx_map[original_global_rel_idx]

                dset_hdf5_img_indices[i] = original_hdf5_img_idx
                dset_global_rel_indices[i] = original_global_rel_idx

                # dset_phrases[i] = item['phrase']

            except Exception as e:
                print(f"Error processing item at index {i}: {e}")
                print(
                    f"  Item details (if available): HDF5 img idx {dataset.rel_to_img_idx_map.get(dataset.filtered_rel_indices[i], 'N/A')}, "
                    f"Global rel idx {dataset.filtered_rel_indices[i]}")
                # Fill with zeros for this problematic sample to keep HDF5 consistent
                dset_global_img[i] = np.zeros(img_feat_dim, dtype=np.float32)
                dset_sub_text[i] = np.zeros(text_feat_dim, dtype=np.float32)
                # ... and so on for all dsets ...
                dset_obj_text[i] = np.zeros(text_feat_dim, dtype=np.float32)
                dset_sub_vis[i] = np.zeros(vis_feat_dim, dtype=np.float32)
                dset_obj_vis[i] = np.zeros(vis_feat_dim, dtype=np.float32)
                dset_union_vis[i] = np.zeros(vis_feat_dim, dtype=np.float32)
                dset_sub_box[i] = np.zeros(box_dim, dtype=np.float32)
                dset_obj_box[i] = np.zeros(box_dim, dtype=np.float32)
                dset_labels[i] = -1  # Special label for error
                dset_hdf5_img_indices[i] = -1
                dset_global_rel_indices[i] = -1
                # dset_phrases[i] = "ERROR_PROCESSING_ITEM"

        # Save some metadata
        out_hf.attrs['description'] = "Precomputed CLIP features for Visual Genome relationships"
        out_hf.attrs['source_hdf5'] = config["hdf5_path"]
        out_hf.attrs['clip_model_type'] = config["clip_model_type"]
        out_hf.attrs['creation_date'] = str(datetime.datetime.now())

        # Save the vocabularies (id2object, id2predicate) from VGDataset for convenience
        # These are string-keyed dicts, so store as JSON strings in attributes or separate datasets
        out_hf.attrs['idx_to_object_name'] = json.dumps(dataset.idx_to_object_name)
        out_hf.attrs['idx_to_predicate_name'] = json.dumps(dataset.idx_to_predicate_name)

        print(f"Preprocessing complete. Features saved to {config['output_hdf5_path']}")

    if hasattr(dataset, 'close'):
        dataset.close()


if __name__ == '__main__':
    # Ensure paths in CONFIG are correct for your system before running!
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(CONFIG["output_hdf5_path"])
    if output_dir:  # Check if output_dir is not an empty string (e.g. for relative path in current dir)
        os.makedirs(output_dir, exist_ok=True)
    # Check if all source files exist
    required_files = [CONFIG["hdf5_path"], CONFIG["dict_json_path"]]  # capgraphs is optional
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"ERROR: Required file not found: {f_path}")
            print("Please check paths in CONFIG.")
            exit(1)
    if not os.path.exists(CONFIG["image_dir"]):
        print(f"ERROR: Image directory not found: {CONFIG['image_dir']}")
        exit(1)

    preprocess_and_save_features(CONFIG)

    # --- Verification (Optional) ---
    print("\n--- Verifying a few entries from the saved HDF5 ---")
    try:
        with h5py.File(CONFIG["output_hdf5_path"], 'r') as verify_hf:
            print("Keys in output HDF5:", list(verify_hf.keys()))
            print("Attributes:", dict(verify_hf.attrs))
            num_saved_samples = verify_hf['global_image_features'].shape[0]
            print(f"Number of samples in saved HDF5: {num_saved_samples}")
            if num_saved_samples > 0:
                print("Sample 0 data:")
                print(f"  Global Img Feat (shape): {verify_hf['global_image_features'][0].shape}")
                print(f"  Sub Text Feat (shape): {verify_hf['subject_text_features'][0].shape}")
                print(f"  Sub Box: {verify_hf['subject_boxes'][0]}")
                print(f"  Label: {verify_hf['predicate_labels'][0]}")
                print(f"  Orig HDF5 Img Idx: {verify_hf['original_hdf5_img_indices'][0]}")
            if num_saved_samples > 100:  # Print another sample
                print("Sample 100 data:")
                print(f"  Global Img Feat (shape): {verify_hf['global_image_features'][100].shape}")
                print(f"  Label: {verify_hf['predicate_labels'][100]}")

    except Exception as e:
        print(f"Error during verification: {e}")
