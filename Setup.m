function [] = Setup()

config = frame_config();

cuda_root = ''; 
% matconvv can automatically guess cuda dir, if not
% please specify the cuda directory

current_dir = pwd();

cd(fullfile(config.matconvv_path, 'matlab'));

if ispc
    cuda_method = 'nvcc';
else
    cuda_method = 'mex';
end

if isempty(config.gpus)
    vl_compilenn('EnableGPU', false);
else
    if isempty(cuda_root)
        vl_compilenn('EnableGPU', true, 'CudaMethod', 'nvcc');
    else
        vl_compilenn('EnableGPU', true, 'CudaRoot', cuda_root, ...
            'CudaMethod', cuda_method);
    end
end

cd(current_dir);