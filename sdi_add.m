function sdi2_1 = sdi_for(sdi2_1, x)
    % 'n' is the number of layers in the network, including the input
    % layer.
    n = numel(sdi2_1.layers);
    sdi2_1.layers{1}.a{1} = x;
    inputmaps = 1;
    
    % For each layer in the network (skipping over the input layer)...
    for l = 2 : n        
        % For conv layers...
        if strcmp(sdi2_1.layers{l}.type, 'conv')
            for j = 1 : sdi2_1.layers{l}.outputmaps
                z = zeros(size(sdi2_1.layers{l - 1}.a{1}) - [sdi2_1.layers{l}.ks - 1, sdi2_1.layers{l}.ks - 1, 0]);
                for i = 1 : inputmaps
                    z = z + convn(sdi2_1.layers{l - 1}.a{i}, sdi2_1.layers{l}.k{i}{j}, 'valid');
                end
                
                % Add the bias term, and apply the sigmoid function.
                % Store the result as outputmap 'j' of layer 'l'.
                sdi2_1.layers{l}.a{j} = sigmoid(z + sdi2_1.layers{l}.b{j});
            end
            inputmaps = sdi2_1.layers{l}.outputmaps;
        
        % For pooling layers...
        elseif strcmp(sdi2_1.layers{l}.type, 'pool')
            % For each input map...
            for j = 1 : inputmaps
                z = convn(sdi2_1.layers{l - 1}.a{j}, ones(sdi2_1.layers{l}.scale) / (sdi2_1.layers{l}.scale ^ 2), 'valid');          
                % Then take only every 'scale'th pixel.
                sdi2_1.layers{l}.a{j} = z(1 : sdi2_1.layers{l}.scale : end, 1 : sdi2_1.layers{l}.scale : end, :);
            end
        end
    end
    % Concatenate all end layer feature maps into vector
    sdi2_1.fv = [];
    
    % For each of the output maps in the final layer...
    for j = 1 : numel(sdi2_1.layers{n}.a)
        % Get the size of output map 'j' for the final layer.
        sa = size(sdi2_1.layers{n}.a{j});
        sdi2_1.fv = [sdi2_1.fv; reshape(sdi2_1.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    sdi2_1.o = sigmoid(sdi2_1.ffW * sdi2_1.fv + repmat(sdi2_1.ffb, 1, size(sdi2_1.fv, 2)));

end
