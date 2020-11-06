function accuracy = compute_accuracy(classProbs)
    testImages = textread('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\testclasses.txt', '%s');
    actualClass = [];
    [c1, c2]= textread('\\smbhome.uscs.susx.ac.uk\gj63\Documents\CV\Animals_with_Attributes2\classes.txt', '%u %s');
    
    loc = 1;    
    for f = 1:length(testImages)
        for e = 1:length(c2)
            %Equate sizes of test images
            if strcmp(testImages(f), c2(e))
                directory = strcat("JPEGImages/", c2(e,1));
                b = dir(fullfile(directory, '*.jpg'));   
                folderlength = numel(b);
                for d = 1:folderlength
                    actualClass(loc) = f;
                    loc = loc + 1;
                end      
            end
        end
    end
       
    correct = 0;
    wrong = 0;
    for i = 1:length(classProbs)
        %Finds the maximum probability
        [~, indx] = max(classProbs(:,i));
        %Checks if correct
        if indx == actualClass(i)
            correct = correct + 1;
        else
            wrong = wrong + 1;
        end
    end
    %Equates accuracy
    accuracy = (correct/(correct+wrong))*100;
    disp(accuracy);
end