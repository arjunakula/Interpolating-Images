function [] = draw_figures(config, syn_mat, iter, mean_img, SSD, Energy, layer)

[I_syn, syn_mat_norm] = convert_syns_mat(config, mean_img, syn_mat);
for i = 1:size(syn_mat_norm, 4)
    imwrite(syn_mat_norm(:,:,:,i), [config.figure_folder, num2str(layer, 'layer_%02d_'), num2str(i, '%03d.png')]);
end
im = im2uint8(I_syn);
[imind,cm] = rgb2ind(im,256);
% save samples
if iter == 1 && layer == 1
    imwrite(imind, cm, [config.Synfolder, 'animation', '.gif'], 'DelayTime', 0.10, 'Loopcount', inf);
else
    imwrite(imind, cm, [config.Synfolder, 'animation', '.gif'], 'WriteMode', 'append', 'DelayTime', 0.10);
end

imwrite(I_syn,[config.Synfolder, num2str(layer, 'layer_%02d_'), num2str(iter, 'dense_original_%04d'), '.png']);
end