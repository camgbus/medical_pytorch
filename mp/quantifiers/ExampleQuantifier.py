import random
from mp.quantifiers.QualityQuantifier import SegImgQualityQuantifier

class ExampleQuantifier(SegImgQualityQuantifier):

    def __init__(self, version='0.0'):
        super().__init__(version)

    def get_quality(self, mask, x=None):
        scores = []
        
        if len(mask) == len(x):
            for img, seg in zip(x, mask):
                print(f"Processing...\n img: {img}\n seg: {seg}")
                s = self.evaluate_image_segmentation(img, seg)
                scores.append(s)

        metric_1 = sum(scores)
        metric_2 = sum(scores) / len(scores) if len(scores) > 0 else 0
        return {'metric_1': metric_1, 'metric_2': metric_2}
        
    def evaluate_image_segmentation(self, segmentation, image):
        return random.randint(0,10)
