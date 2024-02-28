import struct
import gzip
import numpy as np
import math
import sys
import numpy as array_api
sys.path.append("python/")
import needle as ndl
from needle.autograd import Tensor


def parse_mnist(image_filename: str, label_filename: str) -> tuple:
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # get filename
    new_paths = []
    for path in [image_filename, label_filename]:
        new_paths.append(path[: path.find(".gz")])

    # decompressing
    for path in new_paths:
        with gzip.GzipFile(filename=path + ".gz", mode="rb") as uzf:
            with open(file=path, mode="wb") as wf:
                wf.write(uzf.read())
            print("decompression done")

    # reading X
    with open(file=new_paths[0], mode="rb") as uzx:
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
        res_X = res_X / (res_X.max() - res_X.min())
        res_X = res_X + 0.5 - (res_X.min() + (res_X.max() - res_X.min()) / 2)

    # reading y
    with open(file=new_paths[1], mode="rb") as uzy:
        mg_num = struct.unpack(">i", uzy.read(4))[0]
        num_labels = struct.unpack(">i", uzy.read(4))[0]
        print(mg_num, num_labels)

        temp_fmt = ">" + "B" * num_labels
        res_y = np.array(
            struct.unpack(temp_fmt, uzy.read(num_labels)), dtype=np.dtype(np.uint8)
        )

    return (res_X, res_y)


def softmax_loss(Z: Tensor, y_one_hot: Tensor):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    Z_y: Tensor

    # Z_y = ndl.ops.summation(
    #     Z * y_one_hot, axes=(1,)
    # )  # clear useless values & make it a vector
    # Z_sum = ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1,)))
    # res = ndl.ops.summation(Z_sum - Z_y, axes=(0,)) / Z.shape[0]
    # return res

    return (
        ndl.ops.summation(
            ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1,)))
            - ndl.ops.summation(Z * y_one_hot, axes=(1,)),
            axes=(0,),
        )
        / Z.shape[0]
    )


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    print("-" * 25)
    def one_hot_enocoding(array: array_api.ndarray, num_classes: int):
        return (array_api.eye(num_classes)[array])
    
    def norm(x: Tensor):
        return x / x.sum(axes=(1,)).reshape((x.shape[0],1))
        
    # def binary(x: Tensor):
    itr = math.ceil(X.shape[0] / batch)
    
    for i in range(itr):
        
        start = i * batch
        end = (i + 1) * batch
        if (end <= X.shape[0]):
            cur_batch = batch
        else:
            end = X.shape[0]
            cur_batch = end - start
        cur_X = Tensor(X[start : end]) # ne x id
        cur_y = one_hot_enocoding(y[start : end], W2.shape[1]) # ne x nc
        
        Z1 = ndl.ops.relu(cur_X @ W1) # ne x hd
        
        G2 = norm(ndl.ops.exp(Z1 @ W2)) - cur_y # ne x nc
        
        # Iy = array_api.zeros_like(G2) 
        # Iy[np.arange(cur_batch), cur_y] = 1
        # G2 -= Iy
        
        G1 = ndl.ops.binary(Z1) * (G2 @ W2.transpose())
        
        gradient_W1 = 1 / cur_batch * (cur_X.transpose() @ G1)
        gradient_W2 = 1 / cur_batch * (Z1.transpose() @ G2)

        W1 -= lr * gradient_W1
        W2 -= lr * gradient_W2
    return (W1, W2)




def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
