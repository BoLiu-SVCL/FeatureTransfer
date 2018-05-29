clear all;close all;clc;

addpath ~/video-caffe_gpu4/matlab;
caffe.reset_all();

load train_data_shrink.mat;
idx = randperm(length(objs));
feat = features(idx, :)';
cls = objs(idx, :)';
nSample = length(cls);

gpu_id = 0;
solver_path = 'prototxt/cls_solver.prototxt';

caffe.set_mode_gpu();
caffe.set_device(gpu_id);

tic
solver = caffe.Solver(solver_path);

solver.net.blobs('feat').reshape([4096 nSample]);
solver.net.blobs('cls').reshape([1 nSample]);
solver.net.reshape();
solver.net.blobs('feat').set_data(feat);
solver.net.blobs('cls').set_data(cls);
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