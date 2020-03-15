imds = imageDatastore('train', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
h = waitbar(0,'Please wait...');
for i = 1
    [image,info]=readimage(imds,1);
    G = fspecial('gaussian',[9 9],0.2);
    newimage=imfilter(double(image),G);
    imwrite(double(newimage),info.Filename);
    waitbar(i/1)
end

%  i=imread('0001.png');
% G = fspecial('gaussian',[9 9],0.2);
% i2=imfilter(double(i),G);
% figure(1)
% imshow(i)
% figure(2)
% imshow(i2)