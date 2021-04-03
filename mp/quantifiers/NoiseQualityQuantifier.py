import os
import torch
from mp.quantifiers.QualityQuantifier import ImgQualityQuantifier
from mp.utils.lung_captured import whole_lung_captured as LungFullyCaptured

class NoiseQualityQuantifier(ImgQualityQuantifier):
    def __init__(self, device='cuda:0', version='0.0'):
        # Load models
        self.models = dict()
        self.artefacts = ['blur', 'downsample', 'ghosting', 'motion', 'noise', 'spike']
        self.quality_values = [0, 0.25, 0.5, 0.75, 1]
        for artefact in self.artefacts:
            self.models[artefact] = torch.load(os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], artefact, 'model.zip'))
        super().__init__(device, version)

    def get_quality(self, x, path):
        r"""Get quality values for an image representing the maximum intensity of artefacts in it.

        Args:
            x (data.Instance): an instance for a 3D image, normalized so
                that all values are between 0 and 1. An instance (image) in the dataset
                follows the dimensions (channels, width, height, depth), where channels == 1
            path (string): the full path to the file representing x (only used for checking if the 
                           lung is fully captured)

        Returns (dict[str -> float]): a dictionary linking metrices names to float
            quality estimates for each instance in the dataset
        """
        # Calculate metrices
        metrices = dict()
        # Add metric if lung is fully captured in the scan
        discard, _, _ = LungFullyCaptured(path, gpu, cuda)
        metrices['Lung fully captured'] = not discard

        for artefact in self.artefacts:
            # Load model
            model = self.models[artefact]
            model.eval()
            model.to(self.device)
            min_yhat = 1 # Artefact intensity == 0 --> perfect images
            # Do inference
            with torch.no_grad():
                for x_slice in x:
                    yhat = model(x_slice.unsqueeze(0).to(self.device))

                    # Only for 2D models, not necessary for 3D path trained models, since the whole volume will be inputted
                    # ---------------------------------------------------
                    yhat = yhat.cpu().detach()#.numpy()
                    # Transform one hot vector to likert value
                    _, yhat = torch.max(yhat, 1)
                    yhat = self.quality_values[yhat.item()]
                    # Update max intensity value
                    if yhat < min_yhat:
                        min_yhat = yhat
                    # ---------------------------------------------------

            # Add final intensity level to metrics
            metrices[artefact] = min_yhat

        # Return the metrics
        return metrices