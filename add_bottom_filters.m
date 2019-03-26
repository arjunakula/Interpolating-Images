function net = add_bottom_filters(net, layer)
net = frame_net(net, layer);


function net = frame_net(net, layer)

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false;
opts.addrelu = true;

if ismember(1, layer)
    %% layer 1
    layer_name = '1';
    num_in = 3;
    num_out = 20; % number of filters in the first layer
    filter_sz = 15; % the square filter with filter size 15x15 in the 1st layer
    stride = 1; % sub-sampling size for each filter, 1 means in spatial space, each filter is put every two pixels. 
    pad_sz = 2; % pad size for image. For example, image is 224x224, and we add two pixels (padded image is 228x228) to handle boundary conditions.
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad);
end

if ismember(2, layer)
    %% layer2
    layer_name = '2';
    num_in = 20;
    num_out = 10; % 64 40
    filter_sz = 5; %7 size: 34, 5
    stride = 1;%2 , 1
    pad_sz = 2;%ceil(filter_sz/2);
    pad = ones(1,4)*pad_sz;
    net = add_cnn_block(net, opts, layer_name, filter_sz, filter_sz, num_in, num_out, stride, pad) ;
end
