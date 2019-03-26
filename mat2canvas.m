function I_syn = mat2canvas(syn_mat, config, space)

if nargin < 3
    space = 0;
end

sx = config.sx;
sy = config.sy;
dim3 = size(syn_mat, 3);
num_syn = size(syn_mat, 4);
I_syn = zeros(config.nTileRow * config.sy + space * (config.nTileRow-1), ...
    config.nTileCol * config.sx + space * (config.nTileCol-1), dim3, 'single');
k = 1;
for j = 1:config.nTileRow
    for i = 1:config.nTileCol
        I_syn( 1+(j-1)*sy + (j-1) * space : sy+(j-1)*sy + (j-1) * space, 1+(i-1)*sx + (i-1)*space : sx+(i-1)*sx + (i-1) * space, :) = syn_mat(:,:,:,k);
        k = k+1;
        if k > num_syn
            return;
        end
    end
end