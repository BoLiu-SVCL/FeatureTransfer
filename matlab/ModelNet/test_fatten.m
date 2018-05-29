clear all;close all;clc;

addpath ~/video-caffe_gpu4/matlab;
caffe.reset_all();

feat = h5read('features/vgg16_h5/test.h5', '/feat');
cls = h5read('features/vgg16_h5/test.h5', '/cls');
nSample = length(cls)*12;

feat = repmat(feat, [12 1]);
feat = reshape(feat, 4096, nSample);
target = repmat(eye(12), [1 length(cls)]);
pose = repmat([0:11], [1 length(cls)]);
cls = repmat(cls, [12 1]);
cls = reshape(cls, 1, nSample);

gpu_id = 4;
model = 'prototxt/fatten.prototxt';
weight = 'models/fatten.caffemodel';

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

tic
net = caffe.Net(model, weight, 'test');

net.blobs('feat').reshape([4096 nSample]);
net.blobs('target').reshape([12 nSample]);
net.blobs('pose').reshape([1 nSample]);
net.blobs('cls').reshape([1 nSample]);
net.reshape();
net.blobs('feat').set_data(feat);
net.blobs('target').set_data(target);
net.blobs('pose').set_data(pose);
net.blobs('cls').set_data(cls);
toc

tic
net.forward_prefilled();
% fc2 = net.blobs('fc2').get_data();
% fc8 = net.blobs('fc8').get_data();
% [~, pose_pred] = max(fc2);
% [~, cls_pred] = max(fc8);
% pose_pred = pose_pred - 1;
% cls_pred = cls_pred - 1;
% pose_err =  abs(pose_pred - pose);
% cls_err =  abs(cls_pred - pose);
% pose_acc = mean(pose_pred == pose)
% cls_acc = mean(cls_pred == cls)
toc