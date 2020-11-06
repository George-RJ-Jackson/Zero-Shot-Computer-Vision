M = load('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\predicate-matrix-binary.txt');
[c1, c2]= textread('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\classes.txt', '%u %s');
trainImages = textread('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\trainclasses.txt', '%s');
testImages = textread('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\testclasses.txt', '%s');
folderlength = [];
trainfolderlength = [];

%Root
s1 = '\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\JPEGImages\';
trainPred = [];
SVMModel = {};

%Get length of each folder (excluding hidden files)
for a = 1:50  
    directory = strcat("JPEGImages/", c2(a,1));
    b = dir(fullfile(directory, '*.jpg'));
    folderlength(a) = numel(b);
end
disp("Folder length run");
% Set up M for only training predicate
for f = 1:length(trainImages)
    for e = 1:length(c2)
        if strcmp(trainImages(f), c2(e))
            trainPred(f, :) = M(e,:);           
        end
    end
end

% Just get folder length of the training image set
for f = 1:length(trainImages)
    for e = 1:length(c2)
        if strcmp(trainImages(f), c2(e))
            trainfolderlength(f) = folderlength(e);           
        end
    end
end

%Start feature extraction
for iteration = 1:length(trainImages)
    trainImages(iteration) = strcat(s1, trainImages(iteration));
end
    
for iteration = 1:length(testImages)
    testImages(iteration) = strcat(s1, testImages(iteration));
end

%Store images    
imds = imageDatastore(trainImages,'IncludeSubfolders',true,'LabelSource','foldernames');
disp("Imds run");
imdsTest = imageDatastore(testImages,'IncludeSubfolders',true,'LabelSource','foldernames');
disp("imdsTest run");

numTrainImages = numel(imds.Labels);
%Load the network
net = alexnet;
inputSize = net.Layers(1).InputSize; 
%Correct image size
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imds); 
disp("Augment imdsTrain");
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest); 
disp("Augment imdsTest");

%Extract features
layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
disp("featuresTrain run");
featuresTest = activations(net, augimdsTest, layer, 'OutputAs','rows');
disp("featuresTest run");

%Train the models
for l = 1:85
    trainLabels = [];
    loc = 1;
    for k = 1:40
        for j=1:trainfolderlength(k)
            trainLabels(loc) = trainPred(k, l);
            loc = loc + 1;
        end
    end
    disp(l-0.5);
    SVMModel{l} = fitcsvm(featuresTrain,trainLabels);
    disp(l);
    SVMPost{l} = fitSVMPosterior(SVMModel{l});
end

%Begin computation
svmout2 = SVMPost';
attributeProbs = compute_attribute_probs(svmout2, featuresTest);
classProbs = compute_class_probs(attributeProbs);
accuracy = compute_accuracy(classProbs);