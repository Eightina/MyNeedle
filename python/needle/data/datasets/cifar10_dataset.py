import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        super().__init__(transforms)
        self.train = train
        self.p = p
        
        paths = []
        if train:
            for i in range(1, 6):
                pk_path = base_folder + "/data_batch_{}".format(i)
                paths.append(pk_path)
        else:
            paths.append(base_folder + "/test_batch")
        
        X_list = []
        y_list = []
        for path in paths:
            cur_dic = unpickle(path)
            X_list.append(cur_dic[b"data"].reshape((10000, 3, 32, 32)))
            y_list.append(cur_dic[b"labels"])
        self.X = np.concatenate(X_list)
        self.y = np.concatenate(y_list)
            
        metapath = base_folder + "/batches.meta"
        metadic = unpickle(metapath)
        self.y_names = metadic[b"label_names"]
    
    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        (res_X, res_y) = (self.X[index], self.y[index])
        if self.transforms:
            res_X = self.apply_transforms(res_X.reshape((32, 32, 3))).reshape((3, 32, 32))
        return (res_X, res_y)


    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.X)
