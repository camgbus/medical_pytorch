import os
import torch
import mp.utils.load_restore as lr
from mp.quantifiers.QualityQuantifier import ImgQualityQuantifier
from mp.utils.lung_captured import whole_lung_captured as LungFullyCaptured
from mp.utils.create_patches import patchify as Patches

class NoiseQualityQuantifier(ImgQualityQuantifier):
    def __init__(self, output_features, device='cuda:0', version='0.0'):
        # Load models
        self.models = dict()
        self.artefacts = ['blur', 'ghosting', 'motion', 'noise', 'resolution', 'spike']
        self.quality_values = [0, 0.25, 0.5, 0.75, 1]
        for artefact in self.artefacts:
            path_m = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], artefact, 'model_state_dict.zip')
            model = lr.load_model('CNN_Net3D', output_features, path_m, True)
            self.models[artefact] = model
        super().__init__(device, version)

    def get_quality(self, x, path, gpu, cuda):
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
        metrices['LFC'] = not discard

        for artefact in self.artefacts:
            # Load model
            model = self.models[artefact]
            model.to(self.device)
            # Define quality_preds that will include the predictions of each patch per image
            quality_preds = list()
            # Get patches with 50% overlap: (nr_slices, height, width)
            patches = Patches(x, (1, 100, 100, 60), 0.5)
            # Do inference
            with torch.no_grad():
                for patch in patches:
                    yhat = model(patch.unsqueeze(0).to(self.device))
                    yhat = yhat.cpu().detach()
                    # Transform one hot vector to likert value
                    _, yhat = torch.max(yhat, 1)
                    yhat = self.quality_values[yhat.item()]
                    # Add yhat to the running list quality_preds
                    quality_preds.append(yhat)

            # Claculate the mean quality for predictions based on the patches
            metrices[artefact] = sum(quality_preds) / len(quality_preds)

        # Return the metrics
        return metrices