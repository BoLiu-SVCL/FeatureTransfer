clear all;close all;clc;

addpath ~/video-caffe_gpu4/matlab;
caffe.reset_all();

nPose = 13;
nBatch = 2;

load train_data_shrink.mat;
idx = randperm(length(objs));
feat = features(idx, :)';
cls = objs(idx, :)';
nSample = length(cls)*nPose;

feat = repmat(feat, [nPose 1]);
feat = reshape(feat, 4096, nSample);
target = repmat(eye(nPose), [1 length(cls)]);
pose = repmat([0:nPose-1], [1 length(cls)]);
cls = repmat(cls, [nPose 1]);
cls = reshape(cls, 1, nSample);

index = randperm(nSample);
index = reshape(index, nBatch, nSample/nBatch);

gpu_id = 6;
solver_path = 'prototxt/fatten_solver.prototxt';
weight1 = 'models/cls.caffemodel';
weight2 = 'models/pose.caffemodel';

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

tic
solver = caffe.Solver(solver_path);
solver.net.copy_from(weight1);
solver.net.copy_from(weight2);

solver.net.blobs('feat').reshape([4096 nSample/nBatch]);
solver.net.blobs('target').reshape([nPose nSample/nBatch]);
solver.net.blobs('pose').reshape([1 nSample/nBatch]);
solver.net.blobs('cls').reshape([1 nSample/nBatch]);
solver.net.reshape();
toc

tic
train_loss = zeros(1000*nBatch, 1);
pose_loss = zeros(1000*nBatch, 1);
cls_loss = zeros(1000*nBatch, 1);
figure(1);
for m = 1:1%10*nBatch
  i = mod(m-1, nBatch) + 1;
  solver.net.blobs('feat').set_data(feat(:, index(i,:)));
  solver.net.blobs('target').set_data(target(:, index(i,:)));
  solver.net.blobs('pose').set_data(pose(:, index(i,:)));
  solver.net.blobs('cls').set_data(cls(:, index(i,:)));
  
  solver.step(10);
  
  pose_loss(m) = solver.net.blobs('loss1').get_data();
  cls_loss(m) = solver.net.blobs('loss2').get_data();
  train_loss(m) = pose_loss(m) + cls_loss(m);
  
  hold off;
  plot(1:m, train_loss(1:m), 'b');
  hold on;
  plot(1:m, pose_loss(1:m), 'r');
  plot(1:m, cls_loss(1:m), 'k');
  drawnow
end
toc