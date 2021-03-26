import os
import torch
from mp.quantifiers.QualityQuantifier import ImgQualityQuantifier

class NoiseQualityQuantifier(ImgQualityQuantifier):
    def __init__(self, version='0.0'):
        super().__init__(version)

    def get_quality(self, x):
        r"""Get quality values for an image representing the maximum intensity of artefacts in it.

        Args:
            x (numpy.Array): a float64 numpy array for a 3D image, normalized so
                that all values are between 0 and 1. The array follows the
                dimensions (channels, width, height, depth), where channels == 1

        Returns (dict[str -> float]): a dictionary linking metrics names to float
            quality estimates
        """
        metrics = dict()
        artefacts = ['blur', 'downsample', 'ghosting', 'motion', 'noise', 'spike']
        for artefact in artefacts:
            # Load model
            model = torch.load(os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], artefact, 'model.zip'))
            model.eval()
            model.to(self.device)
            max_yhat = 0 # Artefact intensity == 0
            # Do inference
            with torch.no_grad():
                for x_slice in x:
                    x_slice = torch.from_numpy(x_slice)#.permute(2, 0, 1)

                    #TODO: Normalize image ?? --> Ask Camila

                    print(x.slice.size()) # --> Should be (1, 299, 299)
                    yhat = model(x_slice.unsqueeze(0))

                    # Only for 2D models, not necessary for 3D path trained models, since the whole volume will be inputted
                    # ---------------------------------------------------
                    yhat = yhat.cpu().detach().numpy()
                    # Transform one hot vector to likert value
                    yhat = torch.max(yhat, 1)
                    # Update max intensity value
                    if yhat > max_yhat:
                        max_yhat = yhat
                    # ---------------------------------------------------

            # Add final intensity level to metrics
            metrics[artefact] = max_yhat

        # Return the metrics
        return metrics