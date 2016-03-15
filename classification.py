from __future__ import print_function

import os
import glob
import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import caffe
import cv2


caffe_root = "/home/trunia1/dev/lib/caffe/"


def load_network(model_def, model_weights, batch_size, input_dim, print_network_layout=False):

    caffe.set_mode_gpu()

    # Use TEST mode, e.g. don't perform dropout
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    net.blobs['data'].reshape(batch_size, input_dim[0], input_dim[1], input_dim[2])

    if print_network_layout:
        print("Network Layout:")
        for layer_name, blob in net.blobs.iteritems():
            print("  layer_name %s \t %s" % (layer_name, str(blob.data.shape)))

    return net


def imagenet_labels():
    labels_file = caffe_root + "data/ilsvrc12/synset_words.txt"
    if not os.path.exists(labels_file):
        raise ValueError("ImageNet label file does not exist")
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    return labels


def image_transformer(net):

    # Caffe is configured to take images in BGR format
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
    transformer.set_mean('data', mu)                # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BG
    return transformer


def predict_images(net, image_dir):

    # Initialize image transformer
    transformer = image_transformer(net)

    # Fetch ImageNet labels from file
    labels = imagenet_labels()

    # Load image files in directory
    im_files = glob.glob(os.path.join(image_dir, "*.jpg"))

    time_total = 0
    num_processed = 0

    for im_file in im_files:

        t_start = datetime.datetime.now()

        image = caffe.io.load_image(im_file)
        imaged_transformed = transformer.preprocess('data', image)

        # Copy image data to placeholder for net
        net.blobs['data'].data[0,:,:] = imaged_transformed

        # Classification by feed-forward run
        output = net.forward()

        # Output probability vector
        output_prob = output['prob'][0]

        t_end = datetime.datetime.now()
        duration = t_end - t_start
        duration = duration.microseconds * 1e-3

        if duration > 0:
            num_processed += 1
            time_total += duration
            time_avg    = time_total / float(num_processed)

        top_predictions = output_prob.argsort()[::-1][:5]
        print("#" * 40)
        print("Classification done in %.1fms [avg = %.1fms]" % (duration, time_avg))
        print("Probabilities and labels: ")
        for i in top_predictions:
            print("  %s (%.2f)" % (labels[i], output_prob[i]))

        #cv2.imshow("Image", image)
        #cv2.waitKey(1)

    cv2.destroyAllWindows()


def visualize_square(data):

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    plt.show()


def visualize_conv_layer(net, layer_name):
    print("Visualizing filters for layer '%s'..." % layer_name)
    filters = net.params[layer_name][0].data
    visualize_square(filters.transpose(0, 2, 3, 1))


if __name__ == '__main__':

    # AlexNet
    #model_def =     "/home/trunia1/dev/lib/caffe/models/bvlc_alexnet/deploy.prototxt"
    #model_weights = "/home/trunia1/data/CaffeModels/bvlc_alexnet.caffemodel"

    # GoogLeNet
    #model_def =     "/home/trunia1/dev/lib/caffe/models/bvlc_googlenet/deploy.prototxt"
    #model_weights = "/home/trunia1/data/CaffeModels/bvlc_googlenet.caffemodel"

    # VGG ILSVR (16 layers)
    model_def =     "/home/trunia1/data/CaffeModels/VGG_ILSVRC_16_layers_deploy.prototxt"
    model_weights = "/home/trunia1/data/CaffeModels/VGG_ILSVRC_16_layers.caffemodel"

    # VGG ILSVR (19 layers)
    #model_def =     "/home/trunia1/data/CaffeModels/VGG_ILSVRC_19_layers_deploy.prototxt"
    #model_weights = "/home/trunia1/data/CaffeModels/VGG_ILSVRC_19_layers.caffemodel"

    # Microsoft ResNets
    model_def =     "/home/trunia1/data/CaffeModels/deep-residual-networks/ResNet-152-deploy.prototxt"
    model_weights = "/home/trunia1/data/CaffeModels/deep-residual-networks/ResNet-152-model.caffemodel"

    #image_dir =     "/home/trunia1/data/MS-COCO/val2014/"
    image_dir = "/home/trunia1/data/test-cat/"

    batch_size = 1
    #input_dim  = (3, 227, 227)     # AlexNet
    input_dim  = (3, 224, 224)      # GoogLeNet, VGG, MS ResNet

    # Load Caffe network from definition+weights
    network = load_network(model_def, model_weights, batch_size, input_dim, False)

    #visualize_conv_layer(network, "conv3")

    # Predict all images in the directory one-by-one
    predict_images(network, image_dir)


