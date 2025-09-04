import os
import torch
import hashlib

class FeatureCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = {}

    def _get_file_path(self, image_id):
        return os.path.join(self.cache_dir, f"{image_id}.pt")

    def _make_key(self, key, **kwargs):
        # Used as the key in dict, ensuring uniqueness but avoiding special characters
        if key in {"sub", "obj", "union"}:
            bbox = kwargs["bbox"]
            bbox_str = "_".join(map(str, map(int, bbox)))
            return f"{key}_{bbox_str}"
        elif "text" in key:
            text = kwargs["text"]
            # Protecting against special characters with hashing
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            return f"{key}_{text_hash}"
        else:
            return key

    def load(self, image_id, key, **kwargs):
        if image_id not in self._cache:
            path = self._get_file_path(image_id)
            if os.path.exists(path):
                self._cache[image_id] = torch.load(path)
            else:
                self._cache[image_id] = {}
        dict_key = self._make_key(key, **kwargs)
        return self._cache[image_id].get(dict_key, None)

    def save(self, feat, image_id, key, **kwargs):
        # Read out dict first (to prevent overwriting), then write
        if image_id not in self._cache:
            path = self._get_file_path(image_id)
            if os.path.exists(path):
                self._cache[image_id] = torch.load(path)
            else:
                self._cache[image_id] = {}
        dict_key = self._make_key(key, **kwargs)
        self._cache[image_id][dict_key] = feat
        # Each save is written back as a whole

        torch.save(self._cache[image_id], self._get_file_path(image_id))
