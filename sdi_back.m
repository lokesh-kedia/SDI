function sdi2_2 = sdi_back(sdi2_2, y)
    n = numel(sdi2_2.layers);

    %   error
    sdi2_2.e = sdi2_2.o - y;
    %  loss function
    sdi2_2.L = 1/2* sum(sdi2_2.e(:) .^ 2) / size(sdi2_2.e, 2);

    %%  backprop deltas
    sdi2_2.od = sdi2_2.e .* (sdi2_2.o .* (1 - sdi2_2.o));   %  output delta
    sdi2_2.fvd = (sdi2_2.ffW' * sdi2_2.od);              
    if strcmp(sdi2_2.layers{n}.type, 'conv')         
        sdi2_2.fvd = sdi2_2.fvd .* (sdi2_2.fv .* (1 - sdi2_2.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(sdi2_2.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(sdi2_2.layers{n}.a)
        sdi2_2.layers{n}.d{j} = reshape(sdi2_2.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(sdi2_2.layers{l}.type, 'conv')
            for j = 1 : numel(sdi2_2.layers{l}.a)
                sdi2_2.layers{l}.d{j} = sdi2_2.layers{l}.a{j} .* (1 - sdi2_2.layers{l}.a{j}) .* (expand(sdi2_2.layers{l + 1}.d{j}, [sdi2_2.layers{l + 1}.scale sdi2_2.layers{l + 1}.scale 1]) / sdi2_2.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(sdi2_2.layers{l}.type, 'pool')
            for i = 1 : numel(sdi2_2.layers{l}.a)
                z = zeros(size(sdi2_2.layers{l}.a{1}));
                for j = 1 : numel(sdi2_2.layers{l + 1}.a)
                     z = z + convn(sdi2_2.layers{l + 1}.d{j}, rot180(sdi2_2.layers{l + 1}.k{i}{j}), 'full');
                end
                sdi2_2.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(sdi2_2.layers{l}.type, 'conv')
            for j = 1 : numel(sdi2_2.layers{l}.a)
                for i = 1 : numel(sdi2_2.layers{l - 1}.a)
                    sdi2_2.layers{l}.dk{i}{j} = convn(flipall(sdi2_2.layers{l - 1}.a{i}), sdi2_2.layers{l}.d{j}, 'valid') / size(sdi2_2.layers{l}.d{j}, 3);
                end
                sdi2_2.layers{l}.db{j} = sum(sdi2_2.layers{l}.d{j}(:)) / size(sdi2_2.layers{l}.d{j}, 3);
            end
        end
    end
    sdi2_2.dffW = sdi2_2.od * (sdi2_2.fv)' / size(sdi2_2.od, 2);
    sdi2_2.dffb = mean(sdi2_2.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
