function imgCell = read_images(config, net)

img_file = [config.working_folder, 'images.mat'];
files = dir([config.inPath '*.jpg']);

if isempty(files)
   files = dir([config.inPath '*.JPEG']); 
end
if isempty(files)
   files = dir([config.inPath '*.png']); 
end

if isempty(files)
    fprintf('error: No training images are found\n');
    keyboard;
end

numImages = 0;

if exist(img_file, 'file')
    load(img_file);
    numImages = length(imgCell);
end

if numImages ~= length(files)
    imgCell = cell(1, length(files));
    for iImg = 1:length(files)
        fprintf('read and process images %d / %d\n', iImg, length(files))
        img = single(imread(fullfile(config.inPath, files(iImg).name)));
        img = imresize(img, net.normalization.imageSize(1:2));
        imgCell{iImg} = img - net.normalization.averageImage;
    end
    save(img_file, 'imgCell');
end