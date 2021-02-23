## file for functions used in density estimation 
import numpy as np 
from mp.utils.Iterators import Dataset_Iterator


def get_intensities(list_of_paths, min_size=100):
        '''goes through the given directories and there through every image-segmentation
        pair, in order to sample intensity values from every consolidation bigger 
        then min_size. 
        Assumes, that images have endings as in UK_Frankfurt.

        Args :
                list_of_paths (list(strings)): every string is a path to a directory we want to get intensity values from

        Returns: (ndarray(floats)): a one-dim array of intensity values
        '''
        list_intesities = []
        for path in list_of_paths:
                if 'UK_Frankfurt2' in path:
                        mode = 'UK_Frankfurt2'
                else: 
                        mode = 'normal'
                ds_iterator = Dataset_Iterator(path,mode=mode)
                samples = ds_iterator.iterate_components(sample_intensities,
                                                threshold=min_size)
                list_intesities.append(samples)
        arr_intensities = np.array(list_intesities).flatten()
        return arr_intensities

                
def sample_intensities(img,seg,props,number=2000):
        '''samples intesity values from from given component of an img-seg pair
        
        Args:
                img (ndarray): image of intensity values
                seg (ndarray): the respective segmentation mask
                props (list(dict)): the list of the regionprops of the image, 
                        for further documentation see skimage -> regionprops
                number (int): how many samples we want to get
                
        Returns: (list(numbers)): the sampled intensity values'''
                
        coords = props.coords
        intensities = np.array([img[x,y,z] for x,y,z in coords])
        samples = np.random.choice(intensities,number)
        return samples