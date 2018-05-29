clear all;close all;clc;

[gt, ~] = libsvmread('data1');

N = [2127, 922, 3049]

pred = zeros(N(1), 100);
for m = 1:100
  fname = sprintf('outputs/output4_gen_%05d', m);
  pred(:,m) = libsvmread(fname);
end

acc = (pred == repmat(gt, [1 100]));
mean(acc(:))