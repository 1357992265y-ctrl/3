import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_nonorm_transform(resolution):
    nonorm_transform = transforms.Compose(
        [transforms.Resize((resolution, resolution),
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor()])
    return nonorm_transform


class FontDataset(Dataset):
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg

        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        # 定义最大参考数量 (Style 和 Component 共用或分别定义，这里设为 16)
        self.max_refs = 16

        # Component 文件夹
        self.component_root = os.path.join(self.root, self.phase, 'component')

        # --- 加载映射 ---
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))

        self.cr_mapping_path = os.path.join(project_root, "cr_mapping_dynamic_3n.json")
        if not os.path.exists(self.cr_mapping_path):
            self.cr_mapping_path = os.path.join(os.path.dirname(args.data_root), "cr_mapping_dynamic_3n.json")

        self.cr_mapping = self.load_json(self.cr_mapping_path)
        if not self.cr_mapping:
            print(f"Warning: Mapping file not found at {self.cr_mapping_path}")

        self.get_path()

    def load_json(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_path(self):
        self.target_images = []
        self.style_to_images = {}
        self.style_char_map = {}  # 风格索引: Style -> Char -> Path

        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        if not os.path.exists(target_image_dir): return

        for style in os.listdir(target_image_dir):
            style_dir = os.path.join(target_image_dir, style)
            if not os.path.isdir(style_dir): continue

            self.style_char_map[style] = {}
            images_related_style = []

            for img in os.listdir(style_dir):
                img_path = os.path.join(style_dir, img)
                self.target_images.append(img_path)
                images_related_style.append(img_path)
                try:
                    name_part = os.path.splitext(img)[0]
                    if '+' in name_part:
                        parts = name_part.split('+')
                        char = parts[-1]
                        self.style_char_map[style][char] = img_path
                except:
                    pass
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = os.path.basename(target_image_path)

        try:
            name_part = os.path.splitext(target_image_name)[0]
            parts = name_part.split('+')
            style = parts[0]
            content = parts[-1]
        except:
            style, content = "unknown", "unknown"

        # 1. Content Image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        if not os.path.exists(content_image_path):
            content_image_path = content_image_path.replace('.jpg', '.png')

        if os.path.exists(content_image_path):
            content_image = Image.open(content_image_path).convert('RGB')
        else:
            content_image = Image.new('RGB', (128, 128), (255, 255, 255))

        # ================= [修改重点] 2. Style Images (K-Shot Reference) =================
        # 逻辑：查找 cr_mapping 中的参考字，在 TargetImage (当前风格) 中找到对应图片
        # 目标：输出 Tensor [16, 3, H, W]

        ref_chars = self.cr_mapping.get(content, [])
        current_style_map = self.style_char_map.get(style, {})

        style_img_list = []
        for ref_char in ref_chars:
            if ref_char in current_style_map:
                try:
                    img = Image.open(current_style_map[ref_char]).convert('RGB')
                    if self.transforms:
                        img = self.transforms[1](img)  # 使用 Style Transforms
                    style_img_list.append(img)
                except:
                    pass

        # 兜底逻辑：如果映射表里没找到任何参考字（或者风格包里缺字）
        # 则回退到随机采样 1 张同风格图片，并复制填充
        if not style_img_list:
            if style in self.style_to_images and self.style_to_images[style]:
                # 尽量不选自己
                candidates = self.style_to_images[style]
                if len(candidates) > 1 and target_image_path in candidates:
                    candidates = [c for c in candidates if c != target_image_path]

                fallback_path = random.choice(candidates)
                img = Image.open(fallback_path).convert('RGB')
                if self.transforms:
                    img = self.transforms[1](img)
                style_img_list.append(img)
            else:
                # 实在没有（比如只有一张图），就用目标图自己
                img = Image.open(target_image_path).convert('RGB')
                if self.transforms:
                    img = self.transforms[1](img)
                style_img_list.append(img)

        # 截断与填充 (Padding)
        if len(style_img_list) > self.max_refs:
            style_img_list = style_img_list[:self.max_refs]

        C, H, W = style_img_list[0].shape
        padded_style_images = torch.zeros(self.max_refs, C, H, W)

        for i, img in enumerate(style_img_list):
            padded_style_images[i] = img

        # ==============================================================================

        # 3. Target Image
        target_image = Image.open(target_image_path).convert("RGB")

        # 4. Component Images (Content Stream)
        # 逻辑：从 component 文件夹加载，使用 Content Transforms
        valid_component_imgs = []
        for comp_name in ref_chars:  # 复用 ref_chars 列表
            img_path = None
            potential_paths = [
                os.path.join(self.component_root, f"{comp_name}.png"),
                os.path.join(self.component_root, f"{comp_name}.jpg")
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    img_path = p
                    break

            if img_path:
                try:
                    img = Image.open(img_path).convert('RGB')
                    if self.transforms:
                        img = self.transforms[0](img)  # 使用 Content Transforms
                    valid_component_imgs.append(img)
                except:
                    pass

        if len(valid_component_imgs) > self.max_refs:
            valid_component_imgs = valid_component_imgs[:self.max_refs]

        padded_components = torch.zeros(self.max_refs, C, H, W)
        for i, img in enumerate(valid_component_imgs):
            padded_components[i] = img

        # ---------------------------------------------

        if self.transforms:
            content_image = self.transforms[0](content_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "style_image": padded_style_images,  # 这是一个 [16, 3, 96, 96] 的 Tensor
            "target_image": target_image,
            "component_images": padded_components,
            "num_components": len(valid_component_imgs)
        }

        if self.scr:
            sample["neg_images"] = target_image[None, :, :, :]

        return sample

    def __len__(self):
        return len(self.target_images)