clear all;close all;clc;

addpath ~/video-caffe_gpu4/matlab;

feat = h5read('features/vgg16_h5/test.h5', '/feat');
cls = h5read('features/vgg16_h5/test.h5', '/cls');
nSample = length(cls);

gpu_id = 6;
model = 'prototxt/cls.prototxt';
weight = 'models/cls.caffemodel';

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

tic
net = caffe.Net(model, weight, 'test');

net.blobs('feat').reshape([4096 nSample]);
net.blobs('cls').reshape([1 nSample]);
net.reshape();
net.blobs('feat').set_data(feat);
net.blobs('cls').set_data(cls);
toc

tic
net.forward_prefilled();
fc8 = net.blobs('fc8').get_data();
[~, pred] = max(fc8);
pred = pred - 1;
acc = mean(pred == cls);
toc