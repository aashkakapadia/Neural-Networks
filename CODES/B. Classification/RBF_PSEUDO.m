clear all;
clc;

data = xlsread('ACTREC3D_2.xlsx');
hid = 174;
out = 8;                                 %real classes
[samples, col] = size(data);
tra_samples = round(0.6 * samples);
tes_samples = samples - tra_samples;
tra_data = data(1: tra_samples, :);
tes_data = data(tra_samples+1 : end, :);
fea = col - out;

x = data(:, 1:fea);
sigma = zeros(hid, 1);
eta = 1;

%getting centres by hid-means clustering.....
% [~, c, ~] = Kmeans(tra_data(:, 1:fea), hid, out);
 [~, c] = kmeans(tra_data(:, 1:fea), hid);

%initialising sigma as max dis b/w any two centres divided by sqrt(num of centres)
maxDis = 0;
for i = 1 : hid-1              %change this
    for j = i+1 : hid
      dis = euclidian_distance(c(i, :), c(j, :));
      if dis > maxDis
          maxDis = dis;
      end
    end
end
sigma(:, :)= maxDis / sqrt(hid);


%Training the network..................
phi_tra = zeros(tra_samples, hid);
x = tra_data(:, 1:fea);
trueOp = tra_data(:, fea+1 : col);

for i = 1 : tra_samples
    for j = 1 : hid
        phi_tra(i, j) = exp((-1 * (norm(x(i, :) - c(j, :), 2))^2)/(2 * sigma(j, 1)^2));
        if(isnan(phi_tra(i, j)))
            phi_tra(i, j) = 0;
        end
    end
end
% phi_tra = [phi_tra ones(samples,1)];

%finding w matrix by pseudo inverse.....
w = pinv(phi_tra)* trueOp;
predictedOp = phi_tra * w;

predictedLabel = zeros(tra_samples,1);
trueLabel = zeros(tra_samples,1);
    for dat = 1 : tra_samples
      [~, trueLabel(dat)] = max(trueOp(dat, :));
      [~,predictedLabel(dat)] = max(predictedOp(dat, :));
    end

res_tra = [trueLabel predictedLabel]; 
[traEff, tragEff, ~] = efficiency(out, trueLabel, predictedLabel);

%Testing the network...................
phi_tes = zeros(tes_samples, hid);
x = tes_data(:, 1:fea);
trueOp = tes_data(:, fea+1 : col);

for i = 1 : tes_samples
    for j = 1 : hid
        phi_tes(i, j) = exp((-1 * (norm(x(i, :) - c(j, :), 2))^2)/(2 * sigma(j, 1)^2));
        if(isnan(phi_tes(i, j)))
            phi_tes(i, j) = 0;
        end
    end
end

predictedOp = phi_tes * w;
predictedLabel = zeros(tes_samples,1);
trueLabel = zeros(tes_samples,1);
    for dat = 1 : tes_samples
      [~, trueLabel(dat)] = max(trueOp(dat, :));
      [~,predictedLabel(dat)] = max(predictedOp(dat, :));
    end
res_tes = [trueLabel predictedLabel]; 
[tesEff, tesgEff, ~] = efficiency(out, trueLabel, predictedLabel);

%Obtaining results for the provided testing data
x = xlsread('ACTREC3D_test_s2.xlsx');
[res_samples, ~] = size(x); 
phi_res = zeros(res_samples, hid);
for i = 1 : res_samples
    for j = 1 : hid
        phi_res(i, j) = exp((-1 * (norm(x(i, :) - c(j, :), 2))^2)/(2 * sigma(j, 1)^2));
        if(isnan(phi_res(i, j)))
            phi_res(i, j) = 0;
        end
    end
end
result = phi_res * w;
res_label = zeros(res_samples,1);
for dat = 1 : res_samples
      [~, res_label(dat)] = max(result(dat, :));
end