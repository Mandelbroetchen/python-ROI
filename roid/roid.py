from .imset import Imset
from roit.roit import Roit
from pathlib import Path

class Roid:
    def __init__(self, source_path, target_path=None, roit=None):
        
        self.source_path = source_path
        self.source = Imset(source_path)

        if roit is None:
            self.roit = Roit()
        else:
            self.roit = roit

        if target_path is None:
            trans = lambda x: str(x).replace(".", "d")
            suffix = f"{self.roit.maximize}-{trans(self.roit.alpha)}-{trans(self.roit.gamma)}-{self.roit.seed}"
            self.target_path = self.source.root.parent / f"{self.source.root.name}-{suffix}"

        if not Path(self.target_path).exists():
            self.target = None
        else:
            self.target = Imset(self.target_path)

    def transform(self):
        self.target = Imset()
        self.target.root = Path(self.target_path)
        for roi in self.roit.ROI:
            self.roit.roi = roi
            imset_new = self.roit.transform_imset(self.source)
            self.target[imset_new.root.name] = imset_new


    