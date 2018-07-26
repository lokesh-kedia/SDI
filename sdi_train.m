function sdi2 = sdi_train(sdi2, x, y, ptms)

    m = size(x, 3);
    numbatches = m / ptms.batchsize
    
    % Assert that the batch size divides evenly into the number of training
    % examples.
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end   
    sdi2.rL = [];
    
    % For each of the training epochs (one training epoch is one pass over
    % the dataset)...
    for i = 1 : ptms.numepochs
        % Print the current epoch.
        disp(['itirator ' num2str(i) '/' num2str(ptms.numepochs)]);  
        tic;       
        kk = randperm(m);    
        % For each batch...
        for l = 1 : numbatches    
            set_x = x(:, :, kk((l - 1) * ptms.batchsize + 1 : l * ptms.batchsize));
            set_y = y(:,kk((l - 1) * ptms.batchsize + 1 : l * ptms.batchsize));
            % Perform a feed-forward evaluation of the current network on
            % the training batch. 
            sdi2 = sdi_for(sdi2,set_x);
            % Calculate gradients using back-propagation.
            sdi2 = sdi_back(sdi2,set_y); 
            % Update the parameters by applying the gradients.
            sdi2 = sdi_appgrad(sdi2, ptms);
            
            if isempty(sdi2.rL)
                sdi2.rL(1) = sdi2.L;
            end
            sdi2.rL(end + 1) = 0.99 * sdi2.rL(end) + 0.01 * sdi2.L;
        end        
        % Print the elapsed time for this training epoch.
        toc;
    end
    
end
