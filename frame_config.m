function [config, net_cpu] = frame_config(category)
if nargin < 1
   category = ''; 
end

%% GPU setting
% if use gpu, set config.gpus = [1];
% if use cpu, set config.gpus = [];
config.gpus = [];


%% parameters for sampling
% parameter for synthesis
% nTileRow \times nTileCol defines the number of paralle chains
% right now, we currently support square chains, e.g. 2*2, 6*6, 10*10 ...
config.nTileRow = 1; 
config.nTileCol = 1;

% Langevin sampling iteration
config.T = 10;

% standard deviation for reference model q(I/sigma^2)
% no need to change.
config.refsig = 1;
config.Delta = 0.3; 

%% parameters for sampling
% how many layers to learn
config.layer_to_learn = 2;

% learning iterations for each layer
config.nIteration = 700;
% learning rate
config.Gamma = 0.0008;
% batch size, no need to change
config.BatchSize = 32;


%% no need to change
% category name
config.categoryName = category;

% image path: where the dataset locates
config.inPath = ['../Image/', config.categoryName '/'];

% 3rd party path: where the matconvnn locates
config.matconvv_path = '../matconvnet-1.0-beta16/';

% model path: where the deep learning model locates
config.model_path = '../model/';

config.model_name = 'imagenet-vgg-verydeep-16.mat';

run(fullfile(config.matconvv_path, 'matlab', 'vl_setupnn.m'));

net_cpu = load([config.model_path, config.model_name]);
net_cpu = net_cpu.net;

config.sx = net_cpu.normalization.imageSize(1);
config.sy = net_cpu.normalization.imageSize(2);
net_cpu.layers = {};

if isempty(category)
   return; 
end

% result file: no need to change
config.working_folder = ['./working/', '/', config.categoryName, '/'];
config.Synfolder = ['./synthesiedImage/', '/', config.categoryName,  '/'];
config.figure_folder = ['./figure/', '/', config.categoryName, '/'];


% create directory
if ~exist('./working/', 'dir')
    mkdir('./working/')
end

if ~exist('./working/', 'dir')
   mkdir('./working/') 
end

if ~exist('./synthesiedImage/', 'dir')
   mkdir('./synthesiedImage/') 
end

if ~exist('./synthesiedImage/', 'dir')
   mkdir('./synthesiedImage/') 
end

if ~exist('./figure/', 'dir')
   mkdir('./figure/') 
end

if ~exist('./figure/', 'dir')
   mkdir('./figure/') 
end

if ~exist(config.Synfolder, 'dir')
   mkdir(config.Synfolder);
end

if ~exist(config.working_folder, 'dir')
    mkdir(config.working_folder);
end

if ~exist(config.figure_folder, 'dir')
    mkdir(config.figure_folder);
end

