% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc

tic
% Load the training data..................................................
data = xlsread('ACTREC3D_2.xlsx');
% data = load('iris.data');
out = 8;
[samples, col] = size(data);
hid = 200;
fea = col - out;            % No. of input neurons
lam = 1e-4;       % Learning rate
epo = 5000;
epocx = zeros(epo, 2);

% Initialize the weights..................................................
Wi = (rand(hid,fea)*2.0-1.0);  % input weights
Wo = (rand(out,hid)*2.0-1.0);  % Output weights
tra_samples = round(0.6 * samples);
tes_samples = samples - tra_samples;
PreLab = zeros(tra_samples, 1);
TruLab = zeros(tra_samples, 1);

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,fea);
    DWo = zeros(out,hid);
    for sa = 1 : tra_samples
        xx = data(sa,1:fea)';     % Current Sample
        tt = data(sa,fea+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        sign = (Yo .* tt) < 1 ;
        er = tt - Yo;               % Error
        er = er .* sign;
        DWo = DWo + lam * (er * Yh');    % update rule for output weight
        DWi = DWi + lam * ((Wo' *er).*Yh.*(1-Yh))*xx';    %update for feaut weight
        sumerr = sumerr + sum(er.^2);         
    end
    Wi = Wi + DWi;
    Wo = Wo + DWo;
    Deviation = sqrt(sumerr/tra_samples);
    disp(Deviation);
    epocx(ep,:) = [ep Deviation]; 
end
figure(1)
plot(epocx(500:end,1), epocx(500:end,2));
title('Error corresponding to epocx'); xlabel('Epocx'); ylabel('RMS error');

% Validate the network.....................................................
for sa = 1: tra_samples
        xx = data(sa,1:fea)';     % Current Sample
        tt = data(sa,fea+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output 
        [~, PreLab(sa)] = max(Yo(:,1)); %Predicted label
        [~, TruLab(sa)] = max(tt(:,1)); 
             
end
[traEff, ~, ~] = efficiency(out, TruLab, PreLab);

figure(2)
subplot(2,1,1);
scatter3(data(1:tra_samples,1),data(1:tra_samples,2),data(1:tra_samples,3),10, TruLab * 12, 'filled');title('Original clusters');
subplot(2,1,2);
scatter3(data(1:tra_samples,1),data(1:tra_samples,2), data(1:tra_samples,3), 10, PreLab*12, 'filled');title('Predicted clusters');

% Test the network.........................................................
tesData = data(tra_samples+1 : end, :);
Yo = zeros(length(tesData), 1);
PreLab = zeros(tes_samples, 1);
TruLab = zeros(tes_samples, 1);
for sa = 1: tes_samples
        xx = tesData(sa,1:fea)';     % Current Sample
        tt = tesData(sa,fea+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        [~, PreLab(sa)] = max(Yo(:,1)); %Predicted label
        [~, TruLab(sa)] = max(tt(:,1));     
              
end
[tesEff, ~, ~]= efficiency(out, TruLab, PreLab);

figure(3) 
subplot(2,1,1);
scatter3(tesData(:,1),tesData(:,2),tesData(:,3),10, TruLab * 12, 'filled');title('Original clusters');
subplot(2,1,2);
scatter3(tesData(:,1),tesData(:,2), tesData(:,3), 10, PreLab*12, 'filled');title('Predicted clusters');

%Obtaining results for the provided testing data
resData = xlsread('ACTREC3D_test_s2.xlsx');
[res_samples, ~] = size(resData); 
result = zeros(res_samples, 1);
for sa = 1: res_samples
        xx = resData(sa,1:fea)';      % Current Sample
        Yh = 1./(1+exp(-Wi*xx));      % Hidden output
        Yo = Wo*Yh;               % Predicted output
        [~, result(sa)] = max(Yo(:,1)); %Predicted label
end