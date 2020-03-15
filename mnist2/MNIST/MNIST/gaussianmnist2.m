imds = imageDatastore('Test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
h = waitbar(0,'Please wait...');
for i = 1:10000
    [image,info]=readimage(imds,i);
    G = fspecial('gaussian',[9 9],0.2);
    newimage=imfilter(double(image),G);
    imwrite(double(newimage),info.Filename);
    waitbar(i/60000)
end