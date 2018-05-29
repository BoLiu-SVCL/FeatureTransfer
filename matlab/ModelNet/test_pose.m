clear all;close all;clc;

addpath ~/video-caffe_gpu4/matlab;

feat = h5read('features/vgg16_h5/test.h5', '/feat');
pose = h5read('features/vgg16_h5/test.h5', '/pose');
nSample = length(pose);

gpu_id = 6;
model = 'prototxt/pose.prototxt';
weight = 'models/pose.caffemodel';

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

tic
net = caffe.Net(model, weight, 'test');

net.blobs('feat').reshape([4096 nSample]);
net.blobs('pose').reshape([1 nSample]);
net.reshape();
net.blobs('feat').set_data(feat);
net.blobs('pose').set_data(pose);
toc

tic
net.forward_prefilled();
fc2 = net.blobs('fc2').get_data();
[~, pred] = max(fc2);
pred = pred - 1;
err =  abs(pred - pose);
idx = find(err > 6);
err(idx) = 12 - err(idx);
acc = zeros(7, 1);
for m = 1:7
  acc(m) = sum(err == m-1)/nSample;
end
toc