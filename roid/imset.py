from pathlib import Path
from PIL import Image
import json
from roit.utils import log_time

class Imset(dict):
    _instances = {}
    extensions = (".png", ".jpg", ".jpeg", ".bmp")

    def __new__(cls, path = None, resolution=(512, 512)):
        if path is None:
            return super(Imset, cls).__new__(cls)
        if path in cls._instances:
            return cls._instances[path]
        instance = super(Imset, cls).__new__(cls)
        cls._instances[path] = instance
        return instance

    def __init__(self, path = None, resolution=(512, 512)):
        self.root = Path(path) if path is not None else None
        self.resolution = resolution
        if self.root:
            if self.root.suffix.lower() in self.extensions:
                self[self.root.name] = Image.open(self.root).convert("RGB").resize(self.resolution)
            else:
                self._load_images()

    #@log_time
    def _load_images(self):
        """
        Recursively load images into nested Imset dictionaries.
        Subfolders become Imset instances.
        """
        for path in self.root.iterdir():
            if path.is_dir():
                # Create nested Imset for subfolder
                sub_imset = Imset(path, resolution=self.resolution)
                self[path.name] = sub_imset

            elif path.is_file() and path.suffix.lower() in self.extensions:
                image = Image.open(path).convert("RGB").resize(self.resolution)
                self[path.name] = image

            elif path.is_file() and path.suffix.lower() in (".json", ".JSON"):
                with open(path, "r") as f:
                    self[path.name] = json.load(f)

    def __repr__(self):
        return self._repr(level=0)

    def _repr(self, level):
        indent = "--" * level
        lines = []

        # Count images vs subfolders
        images = [k for k, v in self.items() if not isinstance(v, Imset)]
        folders = [k for k, v in self.items() if isinstance(v, Imset)]

        lines.append(
            f"{indent}Imset('{self.root.name}') "
            f"[{len(images)} images, {len(folders)} folders]"
        )

        # Recurse into subfolders
        for name in folders:
            lines.append(self[name]._repr(level + 1))

        return "\n".join(lines)
    
    #@log_time
    def save(self, path=None):
        """
        Save all images in the Imset to disk, preserving the directory structure.
        """
        if path is None:
            path = self.root
        
        if path is None:
            raise ValueError("Path must be specified for saving Imset.")

        if path.suffix.lower() in self.extensions:
            if self[list(self.keys())[0]].save(path):
                return

        for key, obj in self.items():
            if isinstance(obj, Imset):
                # Create subfolder if it doesn't exist, including parents
                subfolder_path = path / key
                subfolder_path.mkdir(parents=True, exist_ok=True)
                obj.save(subfolder_path)  # Recursively save sub-Imset
            else:
                path.mkdir(parents=True, exist_ok=True)  # Make sure parent exists
                path = path / key
                # Save image to disk
                if isinstance(obj, Image):
                    obj.save(path)
                if isinstance(obj, dict):
                    with open(path, "w") as f:
                        json.dump(obj, f, indent=2)
