import math

def patchify(img, patch_shape, overlap=0.5):
    r"""This function calculates patches of an image with specifed overlap using the specified patch_size.
        Args:
            img: torch tensor with dimensions (depth (nr_slices), channels, height (y), width (x))
            patch_size: tuple consisting of (channels, height (y), width (x), depth (nr_slices)) --> not the same as img dims
            overlap: percentage of the overlap between patches --> if overlap is 1 or negative the function
                     will raise an error, since 100% overlap is not possible --> endless loop.
                     Default: 0.5 for 50% overlap.

        Returns (list): List with all patches whereas each patch is a torch tensor and has the dimension of patch_size
    """
    # Change dimensions so we can work with them
    img = img.permute(1, 0, 2, 3)[0]
    patch_shape = (patch_shape[3], patch_shape[1], patch_shape[2])

    # Define empty list for patches
    patches = list()
    
    # Extract image dimensions
    img_size = img.shape
    
    # Define overlap number
    assert overlap < 1 and overlap >= 0, "Overlap can not be 100% or smaller than 0%." 
    overlap = 100 * overlap
    if overlap == 0:
        overlap = 1
    else:
        overlap = 100 / overlap
    
    # Define steps, patches_idx and helper dictionaries
    step = dict()
    nr_patches = dict()
    patches_idx = dict()
    helper = dict()
    dimensions = ['x', 'y', 'z'] # --> to loop through
    
    # Calculate the steps to skip between slides
    for i, dim in enumerate(dimensions):
        # Calculate step size
        step[dim] = patch_shape[i]//overlap
        assert step[dim] != 0,\
        "Overlap is too small and will cause a division by 0 --> specify higher overlap percentage."

        # Define number of necessary idx --> nurmber of patches per dimension
        nr_patches[dim] = (img_size[i] / step[dim])-1  # How often fits step in img dimension dimension
        # Extract last patch if number of patches in not even
        if not (nr_patches[dim]).is_integer(): # Does not fit whole in dimension
            helper[dim] = (img_size[i] - patch_shape[i], img_size[i])  # Last patch idxs
        # Transform float value to integer
        nr_patches[dim] = math.floor(nr_patches[dim])
    
        # Calculate the start and end idx for the patches
        run_step = 0
        patches_idx[dim] = list()
        for idx in range(0, nr_patches[dim], 2):
            run_step += max(0, (idx-1))*step[dim]    # Start idx for patch
            # (Start, End) idx for patch
            patches_idx[dim].append((int(run_step), int(min(run_step+patch_shape[i], img_size[i]))))
            if run_step+(idx+1)*patch_shape[i] >= img_size[i]:  # If next patch not in range break from loop 
                break
        if helper.get(dim, None) is not None:  
            # Add the last patch if it exists, ie. number of patches was not even
            patches_idx[dim].append(helper[dim])
    
    # Extract the patches based on patch idxs and store in patches list
    for x_idxs in patches_idx['x']:
        for y_idxs in patches_idx['y']:
            for z_idxs in patches_idx['z']:
                patch = img[x_idxs[0]:x_idxs[1], y_idxs[0]:y_idxs[1], z_idxs[0]:z_idxs[1]]
                patch = patch.unsqueeze(0).permute(0, 2, 3, 1)
                patches.append(patch)

    # Return the patches
    return patches