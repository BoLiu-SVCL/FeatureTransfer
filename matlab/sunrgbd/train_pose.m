clear all;close all;clc;

addpath ~/video-caffe_gpu4/matlab;
caffe.reset_all();

load train_data_shrink.mat;
degs = (degs + 90)/15;
idx = randperm(length(objs));
feat = features(idx, :)';
pose = round(degs(idx, :))';
nSample = length(pose);

gpu_id = 0;
solver_path = 'prototxt/pose_solver.prototxt';

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

tic
solver = caffe.Solver(solver_path);

solver.net.blobs('feat').reshape([4096 nSample]);
solver.net.blobs('pose').reshape([1 nSample]);
solver.net.reshape();
solver.net.blobs('feat').set_data(feat);
solver.net.blobs('pose').set_data(pose);
toc

tic
train_loss = zeros(1000, 1);
figure(1);
for m = 1:1000
  solver.step(1);
  
  train_loss(m) = solver.net.blobs('loss').get_data();
  
  hold off;
  plot(1:m, train_loss(1:m), 'b');
  drawnow
end
toc