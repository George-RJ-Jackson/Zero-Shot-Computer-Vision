function probs = compute_attribute_probs(model, featuresTest)    
    for i = 1:size(model, 1) 
        for j = 1:size(featuresTest, 1) 
            disp(i-0.5);
            [label, scores] = predict(model{i}, featuresTest(j,:));
            probs(i, j) = scores(2);
            assert(sum(scores) == 1);
        end
        disp(i);
    end
end

