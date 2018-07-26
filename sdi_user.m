function sdi_user(filename)
  load txx1;
  pkg load image;
  pkg load image;
  img=imread(filename);
  img=rgb2gray(img);
  img=imresize(img,[56 56]);
  testx(1,:)=img(:);
  x=double(reshape(testx',56,56,1))/255;
  x(:,:,2)=txx1;
  load sdi56_test;
  sdi3=sdi;
    sdi3 = sdi_for(sdi3, x);
    % For each example, assign the category with the maximum output score.
    [maxval, h] = max(sdi3.o);
    fprintf('\nYou have  %.2f%% chances of Melanoma(--kind of skin cancer--)\n\n',sdi3.o(1,1)*100);
    fprintf('You have %.2f%% chances of Nevus\n\n',sdi3.o(2,1)*100);
    fprintf('You have %.2f%% chances of Seborrheic_keratosis\n\n',sdi3.o(3,1)*100);

    
    % Convert the test labels from binary vectors into integers.
    %[~, a] = max(y);
    % Find the indeces of all of the incorrectly classified examples.
    %bad = find(h ~= a);
    % Divide the number of incorrect classifications by the number of 
    % test examples to calculate the error.
    %er = numel(bad) / size(y, 2)
    %numRight = size(y, 2) - numel(bad);

%fprintf('Accuracy: %.2f%%\n', numRight / size(y, 2) * 100);
end
