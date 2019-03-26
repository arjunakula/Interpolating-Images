function img_cell = imgmat2cell(img_mat)
img_cell = [];
if isempty(img_mat)
    return;
end
numImages = size(img_mat, 4);

img_cell = cell(numImages, 1);
for i = 1:numImages
    img_cell{i} = img_mat(:,:,:,i);
end