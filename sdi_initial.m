function sdi1 = sdi_initial(sdi1, x, y)
  
    inputmaps = 1;  
    mapsize = size(squeeze(x(:, :, 1)));

    % For each defined layer 'l'
    for l = 1 : numel(sdi1.layers)       
        % If this is a pooling layer...
        if strcmp(sdi1.layers{l}.type, 'pool')
            % Calculate the size of the output map for this layer.
            mapsize = mapsize / sdi1.layers{l}.scale;           
            % Assert that the resulting output map size is an integer.
            %inko bhi hatana hai
            assert(all(floor(mapsize) == mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);          
            for j = 1 : inputmaps
                sdi1.layers{l}.b{j} = 0;
            end
        end       
        % If this is a conv layer...
        if strcmp(sdi1.layers{l}.type, 'conv')
            mapsize = mapsize - sdi1.layers{l}.ks + 1;
            cal_out=sdi1.layers{l}.outputmaps * sdi1.layers{l}.ks ^ 2;
 
            for j = 1 : sdi1.layers{l}.outputmaps  
                cal_in=inputmaps * sdi1.layers{l}.ks ^ 2;
                
                for i = 1 : inputmaps
                    % Randomly initialize the 2D filter.
                    sdi1.layers{l}.k{i}{j} = (rand(sdi1.layers{l}.ks) - 0.5) * 2 * sqrt(6 / (cal_in + cal_out));
                end            
                % Initialize the bias term for this neuron to 0.
                sdi1.layers{l}.b{j} = 0;
            end
            inputmaps = sdi1.layers{l}.outputmaps;
        end
    end
    
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);
    % Initialize the weights for the output neurons.
    sdi1.ffb = zeros(onum, 1);
    sdi1.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
