from PIL \
import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    fe = FeatureExtractor()
    i = 0
    for img_path in sorted(Path("./static/img/18").glob("*.jpg")):
        i += 1
        print(img_path, i, sep='    ')  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature/18") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
