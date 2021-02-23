import sys
import os
from DataConnector import DataConnector
from quantifiers.ExampleQuantifier import ExampleQuantifier

# Structure of dirs:
# /
# |---WORKFLOW_DIR
#     |---OPERATOR_IN_DIR
#     |   |---0001
#     |   |   |---img
#     |   |   |   |---img.nii
#     |   |   |---seg
#     |   |       |---001.nii
#     |   |       |---002.nii
#     |   |       |---...
#     |   |---0002
#     |   |   |---img
#     |   |   |   |---img.nii
#     |   |   |---seg
#     |   |       |---001.nii
#     |   |       |---002.nii
#     |   |       |---...
#     |   |---...
#     |---OPERATOR_OUT_DIR
#     |---OPERATOR_TEMP_DIR

if __name__ == "__main__": 

    # For local testing uncomment and set the following environmental vars.
    # These will later be automatically set by the workflow that triggers the docker container
    
    basedir = os.path.dirname(os.path.realpath(__file__))
    os.environ["WORKFLOW_DIR"] = os.path.join(basedir, 'data_dirs')
    os.environ["OPERATOR_IN_DIR"] = "input"
    os.environ["OPERATOR_OUT_DIR"] = "output"
    os.environ["OPERATOR_TEMP_DIR"] = "temp"
    os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(basedir, 'data_dirs', 'persistent')
    
    # In order to use these directories, always join them to WORKFLOW_DIR    
    # Only exception is the OPERATOR_PERSISTENT_DIR since it probably won't be located inside the WORKFLOW_DIR
    
    # Example how all dirs should be used:
    input_dir = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"])
    output_dir = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"])
    temp_dir = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_TEMP_DIR"])
    persistent_dir = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"])

    # -------------------------------------------------------------------------------------------------------
    
    # The following shows the usage of the DataConnector to obtain images / segmentations
    # from these environ paths. It also provides a method for aggregating the metrics and writing
    # them as a JSON file to the output path.
    
    # You don't actually have to use my DataConnector. It's also okay if you just pass the paths to 
    # your own classes. There are only two points you should adhere:
    # 1. Do not use directories outside the specified ones from above
    # 2. Make sure that after completion there is a file called "metrics.json" inside the OPERATOR_OUT_DIR

    # -------------------------------------------------------------------------------------------------------

    # -------------------------
    # Initialize Data Connector
    # -------------------------
    
    dc = DataConnector(extension="mhd")
    dc.loadData()    
    
    # ---------------------------
    # --- DataConnector Usage ---
    # ---------------------------
    
    # Obtain all segmentation and image paths:    
    seg_path_list = dc.getAllSeg()
    img_path_list = dc.getAllImg()
    
    # Obtain all segmentations and images loaded as ITK:    
    seg_itk_list = dc.getAllSegAsItk()
    img_itk_list = dc.getAllImgAsItk()
    
    # Obtain data paths pairwise:
    for inst in dc.instances:
        seg_path = inst.seg
        img_path = inst.img

    # Obtain data as ITK pairwise:
    for inst in dc.instances:
        seg_itk = inst.getItkSeg()
        img_itk = inst.getItkImg()
    
    # ==================================
    # --- ADD MEASUREMENT CALCS HERE ---
    # ==================================
    
    # Init Quantifier
    exampleQuantifier = ExampleQuantifier(version="1.0")
    
    # Get Data from DataConnector
    segmentations = dc.getAllSeg()
    imgs = dc.getAllImg() 
       
    # Calc Metrics:
    metrics = exampleQuantifier.get_quality(x=imgs, mask=segmentations)  
    
    # Append Metrics:
    dc.appendMetric(metrics)
    dc.appendMetric({'m01': 68.697, 'm02': 'test'})    
    
    # -------------------
    # Create Final Output
    # -------------------
    
    # write all metrics as JSON to the workspace output dir
    dc.createOutputJson()    
    
