
import mindspore as ms
import slidingwindow as sw
import numpy as np
import scipy as sp


class BambooConvolution(ms.nn.Cell):
    def __init__(self, input_channels=3):
        super(BambooConvolution, self).__init__()
        self.conv2d_1 = ms.nn.Conv2d(input_channels, 16, 3, pad_mode='same')
        self.conv2d_2 = ms.nn.Conv2d(16, 32, 3, pad_mode='same')
        self.conv2d_3 = ms.nn.Conv2d(32, 32, 3, pad_mode='same')
        
        self.dense_1 = ms.nn.Dense(2048, 64)
        self.dense_2 = ms.nn.Dense(64, 2)
        
        self.activation = ms.nn.ReLU()
        self.classifier = ms.nn.Sigmoid()
        self.pooling = ms.nn.MaxPool2d(2, 2)
        self.dropout = ms.nn.Dropout(0.9)
        self.flatten = ms.nn.Flatten()
        self.add = ms.ops.Add()
        self.concat = ms.ops.Concat(1)
        
    def construct(
        self, x
    ):
        x = self.conv2d_1(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)
        
        
        x = self.conv2d_2(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)
        
        """
        x2 = self.pooling(x2)
        
        x = self.concat((x, x2))
        x = self.activation(x)
        """
        
        x = self.conv2d_3(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)
        """
        ms.ops.Print()(
        ms.ops.Shape()(x)
        )
        """
        x = self.flatten(x)
        
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.dense_2(x)
        
        return x
Net = BambooConvolution


# bc = BambooConvolution()
# param_dict = ms.load_checkpoint('/home/ascend/Desktop/evaluation/notebooks/something_v1.ckpt')
# ms.load_param_into_net(bc, param_dict)
image_size, slide_size = (64, 0.5)

state = {}

def pre_process(image_id, image):
    image = image.asnumpy()
    image = image.transpose((1, 2, 0))
    
    image = image / 255
    image = image.astype(np.float32)
    patches = []
    windows = sw.generate(image, sw.DimOrder.HeightWidthChannel, image_size, slide_size)
    for i,window in enumerate(windows):
        _img = image[window.indices()]
        patches.append(_img)
    
    patches = np.array(patches)
    
    #check total images, how many images are tiled at height direction, and width direction
    n_total = len(windows)
    _x = 0
    
    for i,window in enumerate(windows):
        if _x != window.x:
            n_x = i
            break
        _x = window.x

    state[image_id] = (n_total, n_x, image.shape[0], image.shape[1])

    patches = ms.Tensor(np.transpose(patches, (0,3,1,2)), ms.float32)
    return patches

def post_process(image_id, prediction):
    n_total, n_x, h, w = state[image_id]
    
    prediction = prediction.asnumpy()
    prediction = sp.special.softmax(prediction, axis=1)
    prediction = np.reshape(prediction[:, 1],(n_total // n_x, n_x))
    result = []
    
    for j, row_probability in enumerate(prediction):
        for i, cell_probability in enumerate(row_probability):
            if cell_probability > 0.92:
                result.append([
                    j * (image_size * slide_size) / w,
                    i * (image_size * slide_size) / h,
                    (j+1) * (image_size * slide_size) / w,
                    (i+1) * (image_size * slide_size) / h,
                    cell_probability / 4,
                    cell_probability / 4,
                    cell_probability / 4,
                    cell_probability / 4
                ])
    
    return result

def saliency_map(
    model,
    image_id, 
    image,
    predictions
):

    return image[0] - 0.5
