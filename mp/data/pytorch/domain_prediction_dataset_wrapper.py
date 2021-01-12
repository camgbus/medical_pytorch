from mp.data.pytorch.pytorch_dataset import PytorchDataset
from mp.data.datasets.dataset import Instance
import copy
import torch


class DomainPredictionDatasetWrapper(PytorchDataset):
    r"""Wraps a PytorchDataset to reuse its instances.x and replacing the labels"""

    def __init__(self, pytorch_ds, target_idx):
        """
        Args:
            pytorch_ds (PytorchSegmentationDataset): the Dataset that need to be wrapped
            target_idx (int): the target idx for domain prediction, corresponding to this dataset
        """

        class Dummy:
            def __init__(self):
                self.instances = pytorch_ds.instances
                self.hold_out_ixs = []

        self.original_ds = pytorch_ds

        # Ugly
        # noinspection PyTypeChecker
        super().__init__(dataset=Dummy(), size=pytorch_ds.size)
        # Copy the predictor, but prevent it from reshaping the prediction
        self.predictor = copy.copy(pytorch_ds.predictor)
        self.predictor.reshape_pred = False

        # Create new target as one hot encoded
        # self.target = torch.zeros((1, target_cnt), dtype=self.instances[0].y.tensor.dtype)
        # self.target[:, target_idx] = 1
        self.target = torch.tensor([target_idx], dtype=self.instances[0].y.tensor.dtype)

        # Modify instances
        self.instances = [Instance(inst.x, self.target, inst.name, inst.class_ix, inst.group_id)
                          for inst in self.instances]

    def get_subject_dataloader(self, subject_ix):
        r"""Get a list of input/target pairs equivalent to those if the dataset
        was only of subject with index subject_ix. For evaluation purposes.
        """
        # Generate the original subject dataloader and replace the target
        subject_dataloader = self.original_ds.get_subject_dataloader(subject_ix)
        return [(x, self.target) for x, _ in subject_dataloader]
