# ------------------------------------------------------------------------------
# Super-classes for quality quantifiers.
# ------------------------------------------------------------------------------

class QualityQuantifier():
    r"""The class all quality quantifiers descend from. A QualityQuantifier is a 
    bundle of quality measures for either images or segmentation masks.

    Args:
        version (str): version of the quality quantifier. This number should be
            increased every time a change is made because the results are stored.
    """
    def __init__(self, device='cuda:0', version='0.0'):
        self.name = self.__class__.__name__
        self.version = version
        # Device where all calculations take place. The model and data should be
        # loaded here before performing inference. This is defined by @Simon.
        self.device = device #'cuda:0'
        # All models and information that is statically stored should be placed
        # here. This is defined by @Simon.
        # self.storage_path = storage_path

class ImgQualityQuantifier(QualityQuantifier):
    r"""Super-class for quantifiers of image quality.
    """
    def __init__(self, device='cuda:0', version='0.0'):
        super().__init__(device, version)

    def get_quality(self, x):
        r"""Get quality values for an image according to one or more metrics.
        This method should be overwritten.

        Args:
            x (numpy.Array): a float64 numpy array for a 3D image, normalized so
                that all values are between 0 and 1. The array follows the
                dimensions (channels, width, height, depth), where channels == 1

        Returns (dict[str -> float]): a dictionary linking metric names to float
            quality estimates
        """
        raise NotImplementedError
        return {'metric_1', 0.0, 'metric_2', 0.0}

class SegImgQualityQuantifier(QualityQuantifier):
    r"""Super-class for quantifiers of segmentation quality.
    """
    def __init__(self, version='0.0'):
        super().__init__(version)

    def get_quality(self, mask, x=None):
        r"""Get quality values for a segmentation mask, optionally together with
        an image, according to one or more metrics. This method should be
        overwritten.

        Args:
            mask (numpy.Array): an int32 numpy array for a segmentation mask,
                with dimensions (channels, width, height, depth). The 'channels'
                dimension corresponds to the number of labels, the other
                dimensions should be the same as for x. All entries are 0 or 1.
            x (numpy.Array): a float64 numpy array for a 3D image, normalized so
                that all values are between 0 and 1. The array follows the
                dimensions (channels, width, height, depth), where channels == 1

        Returns (dict[str -> float]): a dictionary linking metric names to float
            quality estimates
        """
        raise NotImplementedError
        return {'metric_1', 0.0, 'metric_2', 0.0}
