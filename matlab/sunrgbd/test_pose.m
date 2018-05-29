clear all;close all;clc;

addpath ~/video-caffe_gpu4/matlab;
caffe.reset_all();

load test_data_shrink.mat;
degs = (degs + 90)/15;
idx = randperm(length(objs));
feat = features(idx, :)';
pose = round(degs(idx, :))';
nSample = length(pose);

gpu_id = 0;
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
acc = zeros(13, 1);
for m = 1:13
  acc(m) = sum(err == m-1)/nSample;
end
toc