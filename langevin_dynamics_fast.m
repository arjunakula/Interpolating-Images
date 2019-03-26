function syn_mat = langevin_dynamics_fast(config, net, syn_mat)
% the input syn_mat should be a 4-D matrix
numImages = size(syn_mat, 4);

if ~isempty(config.gpus)
    net = vl_simplenn_move(net, 'gpu') ;
    dydz = gpuArray(zeros(config.dydz_sz, 'single'));
    syn_mat = gpuArray(syn_mat);
else
    dydz = zeros(config.dydz_sz, 'single');
end

dydz(net.filterSelected) = net.selectedLambdas;
dydz = repmat(dydz, 1, 1, 1, numImages);


for t = 1:config.T
%     fprintf('Langevin dynamics sampling iteration %d\n', t);
    % forward-backward to compute df/dI
    res = vl_simplenn(net, syn_mat, dydz, [], 'conserveMemory', 1, 'cudnn', false);
    
    % part1: derivative on f(I; w)  part2: gaussian I
    syn_mat = syn_mat + config.Delta * config.Delta /2 * res(1).dzdx ...  
        - config.Delta * config.Delta /2 /config.refsig /config.refsig* syn_mat;
    
    % part3: white noise N(0, 1)
    if ~isempty(config.gpus)
        syn_mat = syn_mat + config.Delta * gpuArray(randn(size(syn_mat), 'single'));
    else
        syn_mat = syn_mat + config.Delta * randn(size(syn_mat), 'single');
    end
    clear res;
end

if ~isempty(config.gpus)
    syn_mat = gather(syn_mat);
end
end