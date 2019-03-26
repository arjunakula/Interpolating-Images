function net = train_model_generative(config, net, imdb, getBatch, layer)

rate_list = ones(1,60, 'single');%logspace(-2, -4, 120)*100; %ones(1,60, 'single');
learningRate_array = repmat(rate_list , max(1,floor(config.nIteration / length(rate_list))),1); %logspace(-2, -4, 60) ones(1,60, 'single')
learningRate_array = reshape(learningRate_array, 1, []);

opts.batchSize = config.BatchSize ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = config.gpus; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false;
opts.numFetchThreads = 8;
opts.cudnn = true ;
opts.weightDecay = 0.0001 ; %0.0001
opts.momentum = 0.0 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(config.working_folder, 'matconvnet.bin') ;
opts.learningRate = reshape(learningRate_array, 1, []);

if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

opts.batchSize = min(opts.batchSize, numel(opts.train));
opts.numEpochs = config.nIteration;

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
    for i=1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            J = numel(net.layers{i}.weights) ;
            for j=1:J
                net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end

net.filterSelected = 1:prod(config.dydz_sz);
net.selectedLambdas = ones(1, prod(config.dydz_sz), 'single') * 1;

interval = ceil(opts.numEpochs / 60);

mean_img = net.normalization.averageImage;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus == 1
    gpuDevice(opts.gpus)
end

num_syns = ceil(numel(opts.train) / opts.batchSize) * max(1, numel(opts.gpus));
syn_mats = zeros([config.sx, config.sy, 3, ...
    config.nTileRow * config.nTileCol, num_syns], 'single');

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

for epoch=1:opts.numEpochs
    fprintf('Layer %d / %d, iteration %d / %d\n', layer, config.layer_to_learn, epoch, opts.numEpochs);
    
    learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) * config.Gamma;
    
    % train one epoch and validate
    train = opts.train(randperm(numel(opts.train))) ; % shuffle
    
    [net, syn_mats] = process_epoch_generative(opts, getBatch, epoch, train, learningRate, imdb, net, syn_mats, config);
    
    
    if mod(epoch - 1, interval) == 0 || epoch == opts.numEpochs
        idx_syn = randi(num_syns, 1);
        syn_mat = syn_mats(:,:,:,:, idx_syn);
          
        draw_figures(config, syn_mat, epoch, mean_img, [], [], layer);        
        
        if mod(epoch, inf) == 0 || epoch == opts.numEpochs
            model_file = [config.working_folder, num2str(layer, 'layer_%02d'), '_iter_',...
                num2str(epoch) ,'_model.mat'];
            save(model_file, 'net', 'syn_mats');
        end
    end
end
end
