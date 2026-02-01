import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class CellSegmentationDataset(Dataset):
    """
    Memory-safe dataset for cell segmentation (UNet-friendly).
    """

    def __init__(
        self,
        image_dir: Path,
        annotation_file: Path,
        image_size=(128, 128),
        transform=None
    ):
        # Resolve to absolute paths to avoid issues with DataLoader workers
        self.image_dir = Path(image_dir).resolve()
        annotation_file = Path(annotation_file).resolve()
        self.image_size = image_size
        self.transform = transform

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.images_dict = {img["id"]: img for img in self.coco_data["images"]}

        self.anns_by_image = {}
        for ann in self.coco_data["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Only keep images with annotations AND that exist on disk
        self.image_ids = []
        missing_files = []
        for img_id in self.images_dict:
            if img_id in self.anns_by_image:
                img_path = self.image_dir / self.images_dict[img_id]["file_name"]
                if img_path.exists():
                    self.image_ids.append(img_id)
                else:
                    missing_files.append(self.images_dict[img_id]["file_name"])
        
        # Warn about missing files
        if missing_files:
            print(f"Warning: {len(missing_files)} image(s) referenced in annotations but not found on disk:")
            for fname in missing_files[:10]:  # Show first 10
                print(f"  - {fname}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
            print(f"These images will be skipped. Dataset size: {len(self.image_ids)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images_dict[image_id]
        annotations = self.anns_by_image[image_id]

        # Load image (PIL)
        img_path = self.image_dir / image_info["file_name"]
        image = Image.open(img_path).convert("L")

        # Resize early (VERY IMPORTANT)
        image = image.resize(self.image_size, resample=Image.BILINEAR)

        image = np.array(image, dtype=np.float32)

        # Create mask
        mask = self._create_mask(
            annotations,
            original_size=(image_info["height"], image_info["width"]),
            target_size=self.image_size,
        )

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)

        # Optional transforms (tensor-safe only!)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def _create_mask(self, annotations, original_size, target_size):
        """
        Create binary mask from COCO polygon annotations
        and resize to target size.
        """
        H, W = original_size
        mask_img = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask_img)

        for ann in annotations:
            segmentation = ann.get("segmentation", [])
            if isinstance(segmentation, list):
                for poly in segmentation:
                    if len(poly) >= 6:
                        poly = np.array(poly).reshape(-1, 2)
                        draw.polygon(poly.flatten().tolist(), outline=1, fill=1)

        # Resize mask (nearest to preserve labels)
        mask_img = mask_img.resize(target_size, resample=Image.NEAREST)

        mask = np.array(mask_img, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)  # ensure binary

        return mask
