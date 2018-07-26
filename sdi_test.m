function [er, bad] = sdi_test(sdi3, x, y)
    sdi3 = sdi_for(sdi3, x);
    sdi3.o(:,1:27)
    % For each example, assign the category with the maximum output score.
    [~, h] = max(sdi3.o)  
    % Convert the test labels from binary vectors into integers.
    [~, a] = max(y)
    % Find the indeces of all of the incorrectly classified examples.
    bad = find(h ~= a);
    % Divide the number of incorrect classifications by the number of 
    % test examples to calculate the error.
    er = numel(bad) / size(y, 2)
    numRight = size(y, 2) - numel(bad);

fprintf('Accuracy: %.2f%%\n', numRight / size(y, 2) * 100);
end
