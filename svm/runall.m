clear all;close all;clc;

addpath ~/liblinear/matlab;

nGen = 12;

load ../test_data_1.mat;
load 1-shot-data1.mat;

label_vector_gt = objs + 1;
f = features;
f_norm = sum(abs(f), 2);
f = f ./ repmat(f_norm, [1 4096]);
f_gen = double(features_generated2);
f_norm_gen = sum(abs(f_gen), 2);
f_gen = f_gen ./ repmat(f_norm_gen, [1 4096]);
f_gen = reshape(f_gen', [4096*nGen length(label_vector_gt)])';

% instance_matrix = sparse(f);
% libsvmwrite('data3', label_vector_gt, instance_matrix)

tic
for m = 1:100
%   fprintf('%d\n', m);
%   label_vector = label_vector_gt(rand_idx(m,:));
%   instance_matrix = sparse(f(rand_idx(m,:), :));
%   input_name = sprintf('inputs/input_%05d', m);
%   model_name = sprintf('models/model_%05d', m);
%   output_name = sprintf('outputs/output_%05d', m);
%   libsvmwrite(input_name, label_vector, instance_matrix);
%   comm_train = sprintf('~/liblinear/train -s 3 -c 10 -q -B 1 %s %s', input_name, model_name);
%   comm_predict = sprintf('~/liblinear/predict data1 %s %s', model_name, output_name);
%   system(comm_train);
%   system(comm_predict);
  
  
  fprintf('%d\n', m);
  label_vector = label_vector_gt(rand_idx(m,:));
  label_vector = repmat(label_vector, [1 nGen]);
  label_vector = reshape(label_vector', [10*nGen 1]);
  matrix = f_gen(rand_idx(m,:), :);
  matrix = reshape(matrix', [4096 10*nGen])';
  instance_matrix = sparse(matrix);
  input_name = sprintf('inputs/input4_%05d', m);
  model_name = sprintf('models/model4_%05d', m);
  output_name = sprintf('outputs/output4_gen_%05d', m);
  libsvmwrite(input_name, label_vector, instance_matrix);
  comm_train = sprintf('~/liblinear/train -s 3 -c 10 -q -B 1 %s %s', input_name, model_name);
  comm_predict = sprintf('~/liblinear/predict data1 %s %s', model_name, output_name);
  system(comm_train);
  system(comm_predict);
end
toc
