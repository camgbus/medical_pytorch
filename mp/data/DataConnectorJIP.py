import os
import glob
import json
import SimpleITK as sitk

class DataInstance():    
    r"""A class used to represent data instances that consists of an image and a segmentation
    
    Attributes:
        img (str): path to ITK image file
        seg (str): path to ITK segmentation file
    """
    
    def __init__(self, img_file, seg_file=None):
        self.img = img_file
        self.seg = seg_file
        
    def asItk(self):
        r"""
        Returns:
            (tuple): containing:
                (sitk.Image): image as ITK image
                (sitk.Image): segmentation as ITK image
        """
        return self.getItkImg(), self.getItkSeg()
        
    def getItkImg(self):
        r"""
        Returns:
            (sitk.Image): image as ITK image
        """
        return sitk.ReadImage(self.img)
    
    def getItkSeg(self):
        r"""
        Returns:
            (sitk.Image): segmentation as ITK image
        """
        return sitk.ReadImage(self.seg)

class DataConnector():    
    r"""A class for processing the input data according to the dirs defined in the environmental vars.
    
    Attributes:
        metrics (dict): aggregated storage for all calculated metrics
        instances (List[DataInstance]): a list containing all images and segmentations as type DataInstance 
        ext (str): extension of all images inside the input data dirs.
    """
    
    def __init__(self, extension="mhd"):
        self.metrics = dict()
        self.instances = []
        self.ext = extension
    
    def loadData(self):
        r"""Read all data specified inside the input workflow dir"""
        
        #print(f"Workflow Dir: {os.environ['WORKFLOW_DIR']}")
        #print(f"Op In Dir: {os.environ['OPERATOR_IN_DIR']}")
        #print(f"Op Out Dir: {os.environ['OPERATOR_OUT_DIR']}")
        
        batch_folders = [f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ["OPERATOR_IN_DIR"],'*'))]
        for batch_element_dir in batch_folders:
            img_dir = os.path.join(batch_element_dir, 'img')
            seg_dir = os.path.join(batch_element_dir, 'seg')
            
            if os.path.exists(img_dir):
                img_files = glob.glob(os.path.join(img_dir, f"*.{self.ext}"))
                if len(img_files) > 0:
                    # Assumption: There is only one image and segmentation per instance folder
                    img_file = img_files[0]
                    seg_file = None
                    if os.path.exists(seg_dir):
                        seg_files = glob.glob(os.path.join(seg_dir, f"*.{self.ext}"))
                        if len(seg_files) > 0:
                            seg_file = seg_files[0]
                            self.instances.append(DataInstance(img_file, seg_file))
                    
    def getAllImgAsItk(self):
        r"""
        Returns:
            List[sitk.Image]: a list that contains all read in images loaded as itk.Image
        """
        return [inst.getItkImg() for inst in self.instances]
        
    def getAllSegAsItk(self):
        r"""
        Returns:
            List[sitk.Image]: a list that contains all read in segmentations loaded as itk.Image
        """
        return [inst.getItkSeg() for inst in self.instances]
        
    def getAllImg(self):
        r"""
        Returns:
            List[str]: a list that contains the paths to all input images
        """
        return [inst.img for inst in self.instances]
        
    def getAllSeg(self):
        r"""
        Returns:
            List[str]: a list that contains the paths to all input segmentations
        """
        return [inst.seg for inst in self.instances]
        
    def appendMetric(self, metric):
        r"""Appends a new dictionary of metrics
        
        Args:
              metric (dict): A dictionary that contains some metrics

        Raises:
          TypeError: If the metric is no instance of dict.
        """
        if isinstance(metric, dict):
            self.metrics.update(metric)
        else:
            error = f'Metric must be dict, not {type(metric).__name__}'
            raise TypeError(error)
                    
    def createOutputJson(self):
        r"""Create a JSON file from all the metrics dict and save to the workspace output dir"""
        
        out_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ["OPERATOR_OUT_DIR"])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        out_file = os.path.join(out_dir, 'metrics.json')
        with open(out_file, "w") as fp:
            json.dump(self.metrics , fp)
        
    
