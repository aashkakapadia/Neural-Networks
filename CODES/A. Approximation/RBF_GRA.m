clear all
close all
clc

tic

data = xlsread('fin_2.xlsx');

k = 19;
out = 1;                                 %real classes
[samples, col] = size(data);
tra_samples = round(0.6 * samples);
tes_samples = samples - tra_samples;
fea = col - out;

% Normalising data.............................................
for i = 1 : col
    data(:, i) = (data(:,i) - min(data(:,i))) / (max(data(:, i))- min(data(:,i)));
end
tra_data = data(1: tra_samples, :);
tes_data = data(tra_samples+1 : end, :);

c = zeros(k, fea);
sigma = zeros(k, 1);
eta1 = 1.e-6;
eta2 = 1.e-8;
eta3 = 1.e-8;
x = tra_data(:, 1:fea);

%randomly initialising cluster centres
index = randperm(tra_samples, k);
for run = 1 : k
    c(run, :) = x(index(1, run), :);
end

%initialising sigma as max dis b/w any two centres divided by sqrt(num of centres)
maxDis = 0;
for i = 1 : k-1                           
    for j = i+1 : k
      dis = euclidian_distance(c(i, :), c(j, :));
      if dis > maxDis
          maxDis = dis;
      end
    end
end
sigma(:, 1)= maxDis / sqrt(k);

%Training the network.............
w = 0.1 * rand(k, out);               %weights
trueOp = tra_data(:, col);
phi_tra = zeros(tra_samples, k);
sigMatrix = zeros(k, 1);
disMatrix = zeros(k, tra_samples, fea);

epox = 5000;
epocx = zeros(epox, 2);
for ep = 300 : epox    
 for i = 1 : tra_samples
     for j = 1 : k
         sigMatrix(i, j) = (norm(x(i, :) - c(j, :), 2))^2;
         disMatrix(j, i, :) = x(i, :) - c(j, :);
         phi_tra(i, j) = exp((-1 * sigMatrix(i, j))/(2 * sigma(j, 1)^2));
         if(isnan(phi_tra(i, j)))
             phi_tra(i, j) = 0;
         end
     end
 end
 predictedOp = phi_tra * w;
 err = trueOp - predictedOp;
 
 %updating sigma
 sigma = sigma + eta3 * (w ./ sigma.^3)' * ((phi_tra .* sigMatrix)' * err);

 %updating centres
 for f = 1 : fea
    c(:, f) = c(:, f) - eta2 * (w ./ sigma.^2) .* ((phi_tra .* disMatrix(:, :, f)')' * err);
 end
 
 %updating weights     
 w = w + eta1 * phi_tra' * err;   

 deviation = sqrt(sum(err .^ 2)/tra_samples)
 epocx(ep,:) = [ep deviation];  
 eta1 = eta1 * 1.0009;
 eta2 = eta2 * 1.0009;
 eta3 = eta3 * 1.0009;
end
figure(1)
plot(epocx(:,1), epocx(:,2));title('Epox vs deviation');

%Validate the network
 for i = 1 : tra_samples
     for j = 1 : k
        sigMatrix(i, j) = (norm(x(i, :) - c(j, :), 2))^2;
        phi_tra(i, j) = exp((-1 * sigMatrix(i, j))/(2 * sigma(j, 1)^2));
         if(isnan(phi_tra(i, j)))
             phi_tra(i, j) = 0;
         end
     end
 end
predictedOp = phi_tra * w;
err = trueOp - predictedOp;
deviation = sqrt(sum(err .^ 2)/tra_samples) 
res_tra = [trueOp predictedOp];
disp(['Training error ', num2str(deviation)]);

figure(1)
plot(1:tra_samples, res_tra(:, 1)); %True output
hold on
plot(1:tra_samples, res_tra(:, 2));
title('True output and predicted output for training samples'); xlabel('Sample'); ylabel('Outputs');

%Testing the network..............
phi_tes = zeros(tes_samples, k);
x = tes_data(:, 1:fea);
trueOp = tes_data(:, col);

for i = 1 : tes_samples
    for j = 1 : k
        sigMatrix(i, j) = (norm(x(i, :) - c(j, :), 2))^2;
        phi_tra(i, j) = exp((-1 * sigMatrix(i, j))/(2 * sigma(j, 1)^2));
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

figure(2)
plot(1:tes_samples, res_tes(:,2)); %True output
hold on
plot(1:tes_samples, res_tes(:, 1));
title('True output and predicted output for testing samples'); xlabel('Sample'); ylabel('Outputs');

%Obtaining results on the provided testing data
x = xlsread('fin_test_s2.xlsx');
[res_samples, ~] = size(x); 
result = zeros(res_samples, 1);
phi_res = zeros(res_samples, k);
for i = 1 : res_samples
    for j = 1 : k
        phi_res(i, j) = exp((-1 * (norm(x(i, :) - c(j, :), 2))^2)/(2 * sigma(j,1)^2));
        if(isnan(phi_res(i, j)))
            phi_res(i, j) = 0;
        end
    end
end
result = phi_res * w;
toc