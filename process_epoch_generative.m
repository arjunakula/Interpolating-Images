function  [net_cpu, syn_mats] = process_epoch_generative(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu, syn_mats, config)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net = vl_simplenn_move(net_cpu, 'gpu') ;
else
    net = net_cpu;
end

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 2, mpiprofile on ; end

if numGpus >= 1
    dydz_syn = gpuArray(zeros(config.dydz_sz, 'single'));
else
    dydz_syn = zeros(config.dydz_sz, 'single');
end
dydz_syn(net.filterSelected) = net.selectedLambdas;
dydz_syn = repmat(dydz_syn, 1, 1, 1, config.nTileRow*config.nTileCol);

for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
        ceil(t/opts.batchSize), ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    batchTime = tic ;
    numDone = 0 ;
    res = [] ;
    res_syn = [];

    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        im = getBatch(imdb, batch) ;
        
        if opts.prefetch
            if s==opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            getBatch(imdb, nextBatch) ;
        end
        
        if numGpus >= 1
            im = gpuArray(im) ;
            dydz = gpuArray(zeros(config.dydz_sz, 'single'));
        else
            dydz = zeros(config.dydz_sz, 'single');
        end
        
        % training images
        numImages = size(im, 4);
        dydz(net.filterSelected) = net.selectedLambdas;
        dydz = repmat(dydz, 1, 1, 1, numImages);
        
        cell_idx = (ceil(t / opts.batchSize) - 1) * numlabs + labindex;
        syn_mat = syn_mats(:,:,:,:,cell_idx);
        syn_mat = langevin_dynamics_fast(config, net, syn_mat);
        syn_mats(:,:,:,:,cell_idx) = syn_mat;
        
        res = vl_simplenn(net, im, dydz, res, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', false);
        
        if numGpus >= 1
            syn_mat = gpuArray(syn_mat);
        end
        
        res_syn = vl_simplenn(net, syn_mat, dydz_syn, res_syn, ...
            'accumulate', s ~= 1, ...
            'disableDropout', ~training, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', false);
        numDone = numDone + numel(batch) ;
    end
    
    % gather and accumulate gradients across labs
    if training
        [net, ~] = accumulate_gradients(opts, learningRate, batchSize, net, res, res_syn, config);
    end
    
    clear res;
    clear res_syn;
    
    % print learning statistics
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' [%d/%d]', numDone, batchSize);
    fprintf('\n') ;
end

if numGpus >=1
    net_cpu = vl_simplenn_move(net, 'cpu') ;
else
    net_cpu = net;
end
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, res_syn, config, mmap)
% -------------------------------------------------------------------------
layer_sets = config.layer_sets;
num_syn = config.nTileRow * config.nTileCol;

for l = layer_sets
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        % accumualte from multiple labs (GPUs) if needed
        if nargin >= 8
            tag = sprintf('l%d_%d',l,j) ;
            tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
            for g = setdiff(1:numel(mmap.Data), labindex)
                tmp = tmp + mmap.Data(g).(tag) ;
            end
            res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
            
            if ~isempty(res_syn)
                tag = sprintf('syn_l%d_%d',l,j) ;
                tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
                for g = setdiff(1:numel(mmap.Data), labindex)
                    tmp = tmp + mmap.Data(g).(tag) ;
                end
                res_syn(l).dzdw{j} = res_syn(l).dzdw{j} + tmp ;
            end
        end
        
        if isfield(net.layers{l}, 'weights')
            
            gradient_dzdw = ((1 / batchSize) * res(l).dzdw{j} -  ...
                (1 / num_syn) * res_syn(l).dzdw{j}) / net.numFilters(l);
            if max(abs(gradient_dzdw(:))) > 20 %10
                gradient_dzdw = gradient_dzdw / max(abs(gradient_dzdw(:))) * 20;
            end
            
            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            %             net.layers{l}.momentum{j} = gradient_dzdw;
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR *net.layers{l}.momentum{j};
            
            if j == 1
                res_l = min(l+2, length(res));
                fprintf('\n layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
            end
        end
    end
end
end
 