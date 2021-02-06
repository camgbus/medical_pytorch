## file for functions used in density estimation 
import numpy as np 
from Iterators import Dataset_Iterator


def get_intensities(list_of_paths, min_size=100):
        '''goes through the given directories and there through every image-segmentation
        pair, in order to sample intensity values from every consolidation bigger 
        then min_size. 
        Assumes, that images have endings as in UK_Frankfurt.
        returns: list(floats) a list of intensity values'''
        list_intesities = []
        for path in list_of_paths:
                ds_iterator = Dataset_Iterator(path)
                samples = ds_iterator.iterate_components(sample_intensities,
                                                threshold=min_size)
                list_intesities.append(samples)
        return list_intesities

                
def sample_intensities(img,seg,props,number=2000):
        '''samples intesity values from from given component of an img-seg pair'''
        coords = props.coords
        intensities = np.array([img[x,y,z] for x,y,z in coords])
        samples = np.random.choice(intensities,number)
        return samples