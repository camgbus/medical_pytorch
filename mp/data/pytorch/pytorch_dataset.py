# ------------------------------------------------------------------------------
# This class builds a descendant of torch.utils.data.Dataset from a 
# mp.data.datasets.dataset.Dataset and a list of instance indexes.
# ------------------------------------------------------------------------------

from torch.utils.data import Dataset


class PytorchDataset(Dataset):
    def __init__(self, dataset, ix_lst=None, size=None):
        r"""A dataset which is compatible with PyTorch.

        Args:
            dataset (mp.data.datasets.dataset.Dataset): a descendant of the
                class defined internally for datasets.
            ix_lst (list[int]): list specifying the instances of 'dataset' to 
                include. If 'None', all which are not in the hold-out dataset 
                are incuded.
            size (tuple[int]): desired input size.
        """
        # Indexes
        if ix_lst is None:
            ix_lst = [ix for ix in range(len(dataset.instances))
                      if ix not in dataset.hold_out_ixs]
        self.instances = [ex for ix, ex in enumerate(dataset.instances)
                          if ix in ix_lst]
        self.predictor = None
        self.size = size

    def __len__(self):
        return len(self.instances)

    def get_instance(self, ix=None, name=None):
        r"""Get a particular instance from the ix or name"""
        assert ix is None or name is None
        if ix is None:
            instance = [ex for ex in self.instances if ex.name == name]
            assert len(instance) == 1
            return instance[0]
        else:
            return self.instances[ix]

    def get_ix_from_name(self, name):
        r"""Get ix from name"""
        return next(ix for ix, ex in enumerate(self.instances) if ex.name == name)

    def get_subject_dataloader(self, subject_ix):
        r"""Get a list of input/target pairs equivalent to those if the dataset
        was only of subject with index subject_ix. For evaluation purposes.
        """
        raise NotImplementedError
