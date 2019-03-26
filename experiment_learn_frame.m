function [] = experiment_learn_frame()

close all;
learningTime = tic;
[config, net] = frame_config('rose');

for layer = 1:1:config.layer_to_learn
    
    %% Step 1: add layers
    net = add_bottom_filters(net, layer);
    
    %% Step 2: do some modifications on config
    img = randn(net.normalization.imageSize, 'single');
    if ~isempty(config.gpus)
        gpuDevice(config.gpus(1));
        net = vl_simplenn_move(net, 'gpu') ;
        res = vl_simplenn(net, gpuArray(img));
    else
        res = vl_simplenn(net, img);
    end
    config.dydz_sz = size(res(end).x);
    
    net.numFilters = zeros(1, length(net.layers));
    for l = 1:length(net.layers)
        if isfield(net.layers{l}, 'weights')
            sz = size(res(l+1).x);
            net.numFilters(l) = sz(1) * sz(2);
        end
    end
    config.dydz_sz = size(res(end).x);
    
    config.layer_sets = numel(net.layers):-1:1;
    
    if ~isempty(config.gpus)
        net = vl_simplenn_move(net, 'cpu') ;
    end
    clear res;
    clear img;
    
    %% Step 3 create imdb
    imgCell = read_images(config, net);
    [imdb, getBatch] = convert2imdb(imgcell2mat(imgCell));
    
    %% Step 4: training
    net = train_model_generative(config, net, imdb, getBatch, layer);
end
learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);

%% Step 5: visualize filters
visualize_filters(net);