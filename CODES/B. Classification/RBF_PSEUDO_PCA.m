clear all;
clc;
data = xlsread('BERK7525_2.xlsx');
out = 11;                                 %real classes
[~, col] = size(data);
fea = col - out;             % No. of input neurons
A = data(:, 1 : fea);
[coeff, score] = pca(A);
redFea = 32;
reducedDimension = coeff(:, 1:redFea);
reducedA = A * reducedDimension;
data = [reducedA data(:,fea+1: end)];

hid = 35;                              %real classes
[samples, col] = size(data);
tra_samples = round(0.9 * samples);
tes_samples = samples - tra_samples;
tra_data = data(1: tra_samples, :);
tes_data = data(tra_samples+1 : end, :);
fea = redFea;

% for hid = 162 : 200
    
sigma = zeros(hid, 1);

%getting centres by hid-means clustering.....
% [~, c, ~] = Kmeans_forACT(tra_data(:, 1:fea), hid, out);
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
sigma(:, :)= 2 * maxDis / sqrt(hid);


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
      [~, predictedLabel(dat)] = max(predictedOp(dat, :));
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
%{
hidTra(:, hid) = [hid tragEff tesgEff];
hidTes(:, hid) = [hid traEff tesEff];
end

figure,
plot(hidTra(1, 162:end), hidTra(2, 162:end));
hold on
plot(hidTra(1, 162:end), hidTra(3, 162:end));
legend('Training','Testing');
xlabel('Hidden Neurons');ylabel('Geometric Efficiency');

figure,
plot(hidTes(1, 162:end), hidTes(2, 162:end));
hold on
plot(hidTes(1, 162:end), hidTes(3, 162:end));
legend('Training','Testing');
xlabel('Hidden Neurons');ylabel('Total Efficiency');
%}
%Obtaining results on the provided testing data
x = xlsread('BERK_test_s2.xlsx');

[coeff, score] = pca(x);
reducedDimens = coeff(:, 1:redFea);
x = x * reducedDimens;

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
