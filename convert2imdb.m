function [imdb, fn] = convert2imdb(img_mat)
% --------------------------------------------------------------------
numImages = size(img_mat, 4);
imdb.images.data = img_mat ;
imdb.images.set = ones(1, numImages);
imdb.meta.sets = {'train', 'val', 'test'} ;

fn = @(imdb,batch)getBatch(imdb,batch);
end

function im = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
end