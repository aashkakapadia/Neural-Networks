function [label, c] = Kmeans(data, out, k)
[samples, col] = size(data);
fea = col - k;
x = data(:, 1:fea);

c = zeros(out, fea);
epox = 10;

%randomly initialising cluster centres
index = randperm(samples, out);
for run = 1 : out
    c(run, :) = x(index(1, run), :);
end

tic
%updating cluster centres
for itr = 1 : epox
    label = zeros(samples, 1);
    cen = zeros(out, fea);
    index = zeros(out, fea);
    for i = 1 : samples
        dis = zeros(1, out);
        for d = 1 : out
          dis(1,d) = euclidian_distance(c(d,:), x(i,:));
        end
    [~, I] = min(dis(1,:));
    label(i) = I;
    cen(I, :) = cen(I, :) + x(i, :);
    index(I, :) = index(I, :) + 1;
    end
c = cen ./ index;
end
toc
end