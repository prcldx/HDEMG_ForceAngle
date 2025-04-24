function force = generateForce_ch_0406(spikes,twitch,Tr)
% This script is to generate the channel-wise twitch force
% Input:
% spikes: the firing timings of the channel, 0 and 1 time series
% twitch: the twitch force profile of the channel, assuming same for all
% Tr: rising time of twitch force
% channels

% Output:
% force: the channel-wise motor unit drive

len = length(spikes); % the length of spikes series
firingidx = find(spikes == 1);

if isempty(firingidx)
    force = zeros(1, len);
    return;
end

isi = diff(firingidx);
if isempty(isi)
    isi = inf;
else
    isi(end+1) = isi(end); % 保持长度一致
end

% Gain calculation向量化处理
ratio = Tr ./ isi;
gain = ones(1, length(firingidx));
idx = find(round(Tr/1000*2048)./isi > 0.4);
gain(idx) = (1 - exp(-2 * (ratio(idx)).^3)) ./ ratio(idx);

% Efficient force calculation using matrix addition
twitch_len = length(twitch);
force = zeros(1, len);

% 将所有forcecurve统一叠加处理
for i = 1:length(firingidx)
    start_idx = firingidx(i);
    end_idx = min(start_idx + twitch_len - 1, len);
    range_len = end_idx - start_idx + 1;
    force(start_idx:end_idx) = force(start_idx:end_idx) + gain(i) * twitch(1:range_len);
end
end

