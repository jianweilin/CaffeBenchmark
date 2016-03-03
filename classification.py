from __future__ import print_function

import os
import glob
import datetime

import numpy as np
import caffe
import cv2


caffe_root = "/home/trunia1/dev/lib/caffe/"


def load_network(model_def, model_weights, batch_size):

    caffe.set_mode_gpu()

    # Use TEST mode, e.g. don't perform dropout
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    im_dimension = (224, 244)
    net.blobs['data'].reshape(batch_size, 3, im_dimension[0], im_dimension[1])

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

        duration = datetime.datetime.now() - t_start

        top_predictions = output_prob.argsort()[::-1][:5]
        print("#" * 40)
        print("Classification done in %.1fms" % (duration.microseconds*1e-3))
        print("Probabilities and labels: ")
        for i in top_predictions:
            print("  %s (%.2f)" % (labels[i], output_prob[i]))

        cv2.imshow("Image", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()



if __name__ == '__main__':

    model_def =     "/home/trunia1/dev/lib/caffe/models/bvlc_googlenet/deploy.prototxt"
    model_weights = "/home/trunia1/data/CaffeModels/bvlc_googlenet.caffemodel"
    image_dir =     "/home/trunia1/data/MS-COCO/val2014/"

    batch_size = 1

    # Load Caffe network from definition+weights
    network = load_network(model_def, model_weights, batch_size)

    # Predict all images in the directory one-by-one
    predict_images(network, image_dir)




