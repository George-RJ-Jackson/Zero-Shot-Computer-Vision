function probs = compute_class_probs(attributeProbs)
    [c1, c2]= textread('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\classes.txt', '%u %s');
    testimages = textread('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\testclasses.txt', '%s');
    M = load('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\predicate-matrix-binary.txt');
    trainPred = [];   
    testTotalImages = 1000;
    
    %Preparing predicate matrix for test images
    for f = 1:length(testImages)
        for e = 1:length(c2)
            if strcmp(testImages(f), c2(e))
                trainPred(f, :) = M(e,:);           
            end
        end
    end
    
    for testImage = 1:testTotalImages
        for i = 1:length(testimages) %looping through all test classes
            PClass = [];
            for j = 1:length(trainPred(i,:)) %looping through each attribute in test predicate matrix
                %Compute probs
                if trainPred(i,j) == 0
                    PClass(j) = 1 - attributeProbs(j,testImage);
                else
                    PClass(j) = attributeProbs(j,testImage);
                end
            end
            %Equate product
            product = prod(PClass);
            probs(i, testImage) = product;
        end
    end
end