from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        # get filename
        new_paths = []
        for path in [image_filename, label_filename]:
            new_paths.append(path[:path.find(".gz")])
            
        # decompressing
        for path in new_paths:        
            with gzip.GzipFile(filename=path+".gz", mode='rb') as uzf:
                with open(file=path, mode = "wb") as wf:
                    wf.write(uzf.read())
                print('decompression done')
                
        # reading X      
        with open(file=new_paths[0], mode='rb') as uzx:
            mg_num = struct.unpack(">i", uzx.read(4))[0]
            num_examples = struct.unpack(">i", uzx.read(4))[0]
            height = struct.unpack(">i", uzx.read(4))[0]
            width = struct.unpack(">i", uzx.read(4))[0]
            input_dim = height * width
            print(mg_num, num_examples, height, width, input_dim)
            
            res_X = np.ndarray(shape=(num_examples, input_dim), dtype=np.dtype(np.float32))
            temp_fmt = ">" + "B" * input_dim
            for i in range(num_examples):
                res_X[i] = struct.unpack(temp_fmt, uzx.read(input_dim))
            # normalizing 
            self.images = res_X / 255.0
            
        # reading y
        with open(file=new_paths[1], mode='rb') as uzy:        
            mg_num = struct.unpack(">i", uzy.read(4))[0]
            num_labels = struct.unpack(">i", uzy.read(4))[0]
            print(mg_num, num_labels)
            
            temp_fmt = ">" + "B" * num_labels
            self.labels = np.array(struct.unpack(temp_fmt, uzy.read(num_labels)), dtype=np.dtype(np.uint8))

    def __getitem__(self, index) -> object:
        (res_X, res_y) = (self.images[index], self.labels[index])
        if self.transforms:
            res_X = self.apply_transforms(res_X.reshape((28, 28, -1))).reshape(-1, 28 * 28)
        return (res_X, res_y)

    def __len__(self) -> int:
        return len(self.images)