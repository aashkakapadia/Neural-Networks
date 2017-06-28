% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc

tic
% Load the training data..................................................
data = xlsread('fin_2.xlsx');
[samples, col] = size(data);
out = 1;                    % No. of Output Neurons
fea = col - out;            % No. of feaut neurons
hid = 20;                    % No. of hidden neurons
lam = 1.e-3;                % Learning rate
epo = 5000;
epocx = zeros(epo, 2);

% Normalising data.............................................
for i = 1 : col
    data(:, i) = (data(:,i) - min(data(:,i))) / (max(data(:, i))- min(data(:,i)));
end

% Initialize the weights..................................................
Wi = 0.01 * rand(hid,fea);  % Input weights
Wo = 0.01 * rand(out,hid);  % Output weights
tra_samples = round(0.6 * samples);
tes_samples = samples - tra_samples;

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,fea);
    DWo = zeros(out,hid);
    for sa = 1 : tra_samples
        xx = data(sa,1:fea)';     % Current Sample
        tt = data(sa,fea+1:end)'; % Current Target
%         Yh = 1./(1+exp(-Wi*xx));    % Hidden output using sigmoidal
        Yh = 2./(1+exp(-Wi*xx)) - 1;    % Hidden output using bi-sigmoidal
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        DWo = DWo + lam * (er * Yh'); % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        sumerr = sumerr + sum(er.^2);
        res_tra(sa,:) = [tt Yo];
    end
    Wi = Wi + DWi;
    Wo = Wo + DWo;
    Deviation = sqrt(sumerr/tra_samples);
    disp(Deviation);
    epocx(ep,:) = [ep Deviation];   
end
figure(1)
plot(epocx(:,1), epocx(:, 2));
title('Error corresponding to epocx'); xlabel('Epocx'); ylabel('RMS error');

% Validate the network.....................................................
rmstra = zeros(out,1);
res_tra = zeros(tra_samples, 2);
for sa = 1: tra_samples
        xx = data(sa,1:fea)';       % Current Sample
        tt = data(sa,fea+1:end)';   % Current Target
%         Yh = 1./(1+exp(-Wi*xx));    % Hidden output using sigmoidal
        Yh = 2./(1+exp(-Wi*xx)) - 1;    % Hidden output using bi-sigmoidal
        Yo = Wo*Yh;                 % Predicted output
        rmstra = rmstra + (tt-Yo).^2; 
        res_tra(sa,:) = [tt Yo];
end
disp(['Training error ', num2str(sqrt(rmstra/tra_samples))]);
errr = [hid sqrt(rmstra/tra_samples)];

figure(2)
plot(1:tra_samples, res_tra(:,1)); %True output
hold on
plot(1:tra_samples, res_tra(:, 2));
title('True output and predicted output for training samples'); xlabel('Sample'); ylabel('Outputs');
 
figure(3)
plot(res_tra(:, 1), res_tra(:, 2)); title('True output vs predicted output');
xlabel('True output'); ylabel('Predicted output');
 
 % Test the network.........................................................
rmstes = zeros(out,1);
res_tes = zeros(tes_samples,2);
tesData = data(tra_samples+1 : end, :);
Yo = zeros(length(tesData), 1);

for sa = 1: tes_samples
        xx = tesData(sa,1:fea)';      % Current Sample
        ca = tesData(sa,end);         % Actual Output
%         Yh = 1./(1+exp(-Wi*xx));    % Hidden output using sigmoidal
        Yh = 2./(1+exp(-Wi*xx)) - 1;    % Hidden output using bi-sigmoidal
        Yo(sa) = Wo*Yh;               % Predicted output
        rmstes = rmstes + (ca-Yo(sa)).^2;
        res_tes(sa,:) = [ca Yo(sa)];
end
disp(['Test Error ', num2str(sqrt(rmstes/tes_samples))]);
figure(4)
plot(1:tes_samples, res_tes(:,1)); %True output
hold on
plot(1:tes_samples, res_tes(:, 2));
title('True output and predicted output for testing samples'); xlabel('Sample'); ylabel('Outputs');

%Obtaining results for the provided testing data
resData = xlsread('fin_test_s2.xlsx');
[res_samples, ~] = size(resData); 
result = zeros(res_samples, 1);
for sa = 1: res_samples
        xx = resData(sa,1:fea)';      % Current Sample
%         Yh = 1./(1+exp(-Wi*xx));    % Hidden output using sigmoidal
        Yh = 2./(1+exp(-Wi*xx)) - 1;    % Hidden output using bi-sigmoidal
        Yo(sa) = Wo*Yh;               % Predicted output
        result(sa,:) = Yo(sa);
end
toc