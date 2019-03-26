function net = add_cnn_block(net, opts, id, h, w, in, out, stride, pad, learning_rate)
% --------------------------------------------------------------------
if nargin < 10
    learning_rate = 1;
end

info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
else
    name = 'conv' ;
end
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
    'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(1, out, 'single')}}, ...
    'stride', [stride, stride], ...
    'pad', pad, ...
    'learningRate', learning_rate*[1, 2], ...
    'weightDecay', [opts.weightDecay 0]) ;
if opts.batchNormalization
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',id), ...
        'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
        'learningRate', learning_rate*[2 1], ...
        'weightDecay', [0 0]) ;
end
if opts.addrelu
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id));
end
end


function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    case 'zero'
        weights = zeros(h, w, in, out, type);
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end