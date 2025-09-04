# split_test_to_val.py
import json
import random
import os

def split_dataset(input_json_path, val_ratio=0.2, random_seed=42):
    """
    Split the input scene graph JSON file into new test and validation sets by image ID.

    parameter:
    input_json_path (str): Path to the original test set JSON file.
    val_ratio (float): Ratio of images allocated to the validation set (between 0.0 and 1.0).
    random_seed (int): Random seed for reproducible partitioning.
    """
    print(f"Start processing files: {input_json_path}")
    print(f"Validation set division ratio: {val_ratio*100:.2f}%")

    try:
        with open(input_json_path, 'r') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_json_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{input_json_path}' is not in valid JSON format.")
        return
    except Exception as e:
        print(f"An error occurred while loading file '{input_json_path}': {e}")
        return

    # Setting the random seed
    random.seed(random_seed)

    # 1. Get all image information and scramble it
    all_images_info = original_data['images']
    random.shuffle(all_images_info) # 直接打乱列表

    # 2. Divide image information and image ID
    num_total_images = len(all_images_info)
    num_val_images = int(num_total_images * val_ratio)
    num_new_test_images = num_total_images - num_val_images

    val_images_info = all_images_info[:num_val_images]
    new_test_images_info = all_images_info[num_val_images:]

    val_image_ids = {img['id'] for img in val_images_info}
    new_test_image_ids = {img['id'] for img in new_test_images_info}

    print(f"Total number of original images: {num_total_images}")
    print(f"The number of images allocated to the validation set: {len(val_image_ids)}")
    print(f"The number of images assigned to the new test set: {len(new_test_image_ids)}")

    # 3. filter annotations, instances
    original_annotations = original_data['annotations']
    original_instances = original_data['instances']

    val_annotations = [ann for ann in original_annotations if ann['image_id'] in val_image_ids]
    new_test_annotations = [ann for ann in original_annotations if ann['image_id'] in new_test_image_ids]

    val_instances = [inst for inst in original_instances if inst['image_id'] in val_image_ids]
    new_test_instances = [inst for inst in original_instances if inst['image_id'] in new_test_image_ids]

    print(f"Number of relationship annotations in the validation set: {len(val_annotations)}")
    print(f"Number of relationship annotations in the new test set: {len(new_test_annotations)}")
    print(f"Number of object instances in the validation set: {len(val_instances)}")
    print(f"Number of object instances in the new test set: {len(new_test_instances)}")

    # 4. Prepare new JSON data structure
    val_dataset = {
        'images': val_images_info,
        'annotations': val_annotations,
        'categories': original_data['categories'], 
        'instances': val_instances,
        'relationships': original_data['relationships'] 

    new_test_dataset = {
        'images': new_test_images_info,
        'annotations': new_test_annotations,
        'categories': original_data['categories'],
        'instances': new_test_instances,
        'relationships': original_data['relationships']
    }

    # 5. Save to a new file
    base_dir = os.path.dirname(input_json_path)
    original_filename = os.path.basename(input_json_path)
    name_part, ext_part = os.path.splitext(original_filename)

    val_output_path = os.path.join(base_dir, f"{name_part}_val.json")
    new_test_output_path = os.path.join(base_dir, f"{name_part}_new_test.json")

    try:
        with open(val_output_path, 'w') as f_val:
            json.dump(val_dataset, f_val, indent=2) 
        print(f"验证集已保存到: {val_output_path}")

        with open(new_test_output_path, 'w') as f_new_test:
            json.dump(new_test_dataset, f_new_test, indent=2)
        print(f"新测试集已保存到: {new_test_output_path}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == "__main__":
    # --------------------------------------------------------------------
    # TODO: Users need to set the original test set JSON file path here!
    # --------------------------------------------------------------------
    original_test_json_path = "dataset/vg/annotations/instances_vg_test.json" 

    # You can choose the validation set ratio and random seed
    validation_ratio = 0.6  
    seed_for_splitting = 123 

    if not os.path.exists(original_test_json_path):
        print(f"Error: Input file '{original_test_json_path}' not found. Please check the path.")
    else:
        split_dataset(original_test_json_path,
                      val_ratio=validation_ratio,

                      random_seed=seed_for_splitting)
