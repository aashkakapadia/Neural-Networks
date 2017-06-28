clear all
close all
clc
data = xlsread('fin_2.xlsx');
% data = load('iris.data');
k = 16;
out = 1;                                   %real classes
[samples col] = size(data);
tra_samples = round(0.6 * samples);
tes_samples = samples - tra_samples;
fea = col - out;

%normalizing data
for i = 1 : col
   data(:, i) = (data(:,i) - min(data(:,i))) / (max(data(:, i))- min(data(:,i)));
end
tra_data = data(1: tra_samples, :);
tes_data = data(tra_samples+1 : end, :);

%getting centres by K-means clustering.....
[~, cen] = Kmeans(tra_data, k, out);
% [~, cen] = kmeans(tra_data(:,1:2), k);

%initialising sigma as max dis b/w any two centres divided by sqrt(num of centres)
maxDis = 0;
for i = 1 : k-1                       
    for j = i+1 : k
      dis = euclidian_distance(cen(i, :), cen(j, :));
      if dis > maxDis
          maxDis = dis;
      end
    end
end
sigma = maxDis / sqrt(k);

%Training the network..................
phi_tra = zeros(tra_samples, k);
x = tra_data(:, 1:fea);
trueOp = tra_data(:, col);
for i = 1 : tra_samples
    for j = 1 : k
        phi_tra(i, j) = exp((-1 * (norm(x(i, :) - cen(j, :), 2))^2)/(2 * sigma^2));
        if(isnan(phi_tra(i, j)))
            phi_tra(i, j) = 0;
        end
    end
end

%finding w matrix by pseudo inverse.....
w = pinv(phi_tra)* trueOp;
predictedOp = phi_tra * w;
res_tra = [trueOp predictedOp];
er = (trueOp - predictedOp).^2; 
deviation = sqrt(sum(er)/tra_samples);
disp(['Training error ', num2str(deviation)]);

figure(1)
plot(1:tra_samples, res_tra(:, 1)); %True output
hold on
plot(1:tra_samples, res_tra(:, 2));
title('True output and predicted output for training samples'); xlabel('Sample'); ylabel('Outputs');
figure(2)
plot(res_tra(:, 1), res_tra(:, 2)); title('True output vs predicted output');
xlabel('True output'); ylabel('Predicted output');

%Testing the network...................
phi_tes = zeros(tes_samples, k);
x = tes_data(:, 1:fea);
trueOp = tes_data(:, col);

for i = 1 : tes_samples
    for j = 1 : k
        phi_tes(i, j) = exp((-1 * (norm(x(i, :) - cen(j, :), 2))^2)/(2 * sigma^2));
        if(isnan(phi_tes(i, j)))
            phi_tes(i, j) = 0;
        end
    end
end

predictedOp = phi_tes * w;
res_tes = [trueOp predictedOp];
er = (trueOp - predictedOp).^2; 
deviation = sqrt(sum(er)/tes_samples);
disp(['Testing error ', num2str(deviation)]);

figure(3)
plot(1:tes_samples, res_tes(:,1)); %True output
hold on
plot(1:tes_samples, res_tes(:, 2));
title('True output and predicted output for testing samples'); xlabel('Sample'); ylabel('Outputs');

%Obtaining results on the provided testing data
x = xlsread('fin_test_s2.xlsx');
x = (x - min(x)) / (max(x)- min(x));


[res_samples, ~] = size(x); 
phi_res = zeros(res_samples, k);
for i = 1 : res_samples
    for j = 1 : k
        phi_res(i, j) = exp((-1 * (norm(x(i, :) - cen(j, :), 2))^2)/(2 * sigma^2));
        if(isnan(phi_res(i, j)))
            phi_res(i, j) = 0;
        end
    end
end
result = phi_res * w;