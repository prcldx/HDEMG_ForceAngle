function [Pulses, IPTs, estimationParameters, aveCs,pNum] = winDecomp0406(winData, estimationParameters)
% -- Created by CC -- Email: cedric_c@126.com %
% This function decompose the EMG signals in a sliding window in real time.

exFactor = estimationParameters.exFactor;
centroids = estimationParameters.Centroids;

winLen = size(winData,2); % estimationParameters.winLen;
munum = estimationParameters.munum;

tmpeSig = extend(winData,exFactor);

Pulses = cell(1,munum);
IPTs = zeros(munum,winLen);
aveCs = zeros(munum,1);
W = estimationParameters.W;

    
pNum = zeros(1,munum);
tmpIPTs = W'*tmpeSig;
for mu = 1:munum
    tmpT = tmpIPTs(mu,:);
    tmpT([1:exFactor,end-exFactor+1:end]) = 0;
    validIdx = (exFactor+1):(size(tmpT,2)-exFactor);
    tmpT_center = tmpT(validIdx);
    % 有符号平方
    tT = tmpT_center.^2;
    if -min(tmpT_center) > max(tmpT_center)
        tT(tmpT_center > 0) = 0;
    else
        tT(tmpT_center < 0) = 0;
    end

%     tT = abs(tmpT).*tmpT;
%     if  -min(tmpT) > max(tmpT)
%         tT(tmpT>0) = 0;
%     else
%         tT(tmpT<0) = 0;
%     end
%     tT = abs(tT);

    C1 = centroids(mu,1);
    C2 = centroids(mu,2);
    aveC = (C1+C2)/2;
    tmpInd = find(tT(exFactor+1:end-exFactor)>aveC);
    compInd = tmpInd;

%     intervalLimit = 20;
%     compInd = remRepeatedInd(tT(exFactor+1:end-exFactor),tmpInd,intervalLimit);
    compInd = compInd+exFactor;

    Pulses{mu} = compInd;         
end

