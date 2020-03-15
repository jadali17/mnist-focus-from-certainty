layers = [
    imageInputLayer([28 28 1],"Name","imageinput","Normalization","zscore")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding",[1 1 1 1])
    reluLayer("Name","relu_1")
    batchNormalizationLayer("Name","batchnorm_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_2","Padding",[1 1 1 1])
    reluLayer("Name","relu_2")
    batchNormalizationLayer("Name","batchnorm_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_3","Padding",[1 1 1 1])
    reluLayer("Name","relu_3")
    batchNormalizationLayer("Name","batchnorm_3")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

%Split the data into 70% training and 30% validation
imds = imageDatastore('train', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);






 numClasses = numel(categories(imdsTrain.Labels));

%% network options
miniBatchSize = 10;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
     'ExecutionEnvironment', 'gpu',...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

% net = trainNetwork(imdsTrain,layers,options);%initial
%net2=trainNetwork(imdsTrain,layers,options);%batches before relu
% net3=trainNetwork(imdsTrain,layers,options);%batches after relu
% net4=trainNetwork(imdsTrain,layers,options);%batches after relu and before softmax
net2=trainNetwork(imdsTrain,layers,options);%with adam and batches after relu

%% Supporting functions
imdstest = imageDatastore('Test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[YPred2,probs2] = classify(net2,imdstest);
accuracy = mean(YPred2 == imdstest.Labels)
%  idx = randperm(numel(imdstest.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imdstest,idx(i));
%     imshow(I)
%     label = YPred2(idx(i));
%     title(string(label) + ", " + num2str(100*max(probs2(idx(i),:)),3) + "%");
% end% 
% accuracy = mean(YPred2 == imdsValidation.Labels)
% 
% 
%  plotconfusion(imdstest.Labels,YPred2)%plot confusion matrix
% 
% %display images that were misclassifed
j=1
c=1
for i=1:3000
if(imdstest.Labels(i)~=YPred2(i))
subplot(2,2,j)
I = readimage(imdstest,i);
imshow(I)
label = YPred2(i);
actuallabel=imdstest.Labels(i);
title("Actual label:"+string(actuallabel) +" predicted label:" +string(label) + ", " + num2str(100*max(probs2(i,:)),3) + "%")
j=j+1;
c=c+1
end
if c==4
break
end
end
