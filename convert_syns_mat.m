function [I_syn, syn_mat] = convert_syns_mat(config, mean_img, syn_mat)

space = 5;

color = 0;

for i = 1:size(syn_mat, 4)
    syn_mat(:,:,:,i) = uint8(syn_mat(:,:,:,i) + mean_img);
    syn_mat(:,:,:,i) = single(syn_mat(:,:,:,i));
    gLow = min( reshape(syn_mat(:,:,:,i), [],1));
    gHigh = max(reshape(syn_mat(:,:,:,i), [],1));
    syn_mat(:,:,:,i) = (syn_mat(:,:,:,i)-gLow) / (gHigh - gLow);
end

I_syn = mat2canvas(syn_mat, config, space);


for row = 1:config.nTileRow
    I_syn(row * config.sx + (row-1) * space + 1:row * config.sx + (row-1) * space + space, :, :) = color;
end

for col = 1:config.nTileCol
    I_syn(:, col * config.sy + (col-1) * space + 1:col * config.sy + (col-1) * space + space, :) = color;
end
end