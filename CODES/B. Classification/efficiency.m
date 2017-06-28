function [totalAcr geoAcr averageAcr classAcr] = efficiency(out, trueOp, predictedOp) 

confMatrix = zeros(out,out);
%making confussion matrix
 for i = 1 : length(trueOp)
   confMatrix(trueOp(i), predictedOp(i)) = confMatrix(trueOp(i), predictedOp(i)) + 1;
 end
confMatrix

classAcr = zeros(1, out); 
totalAcr = 0;
averageAcr = 0;
%calculating accuracy
for p = 1 : out
   classAcr(1, p) = confMatrix(p, p) / sum(confMatrix(p,:)); 
   if isnan(classAcr(1, p))
       classAcr(1, p) = 1;
   end
   totalAcr = totalAcr + confMatrix(p, p);
   averageAcr = averageAcr + classAcr(1, p);
end
geoAcr = sqrt(prod(classAcr));
totalAcr = totalAcr / length(trueOp);
averageAcr = averageAcr / out;

classAcr
totalAcr
geoAcr
averageAcr
