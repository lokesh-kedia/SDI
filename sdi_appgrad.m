function sdi2_3 = sdi_appgrad(sdi2_3, pmts)
    for l = 2 : numel(sdi2_3.layers)
        if strcmp(sdi2_3.layers{l}.type, 'conv')
            for j = 1 : numel(sdi2_3.layers{l}.a)
                for ii = 1 : numel(sdi2_3.layers{l - 1}.a)
                    sdi2_3.layers{l}.k{ii}{j} = sdi2_3.layers{l}.k{ii}{j} - pmts.alpha * sdi2_3.layers{l}.dk{ii}{j};
                   % size(sdi2_3.layers{l}.k{ii}{j})
                    %size(sdi2_3.layers{l}.dk{ii}{j})
                    %pause;
                end
                sdi2_3.layers{l}.b{j} = sdi2_3.layers{l}.b{j} - pmts.alpha * sdi2_3.layers{l}.db{j};
            end
        end
    end
    sdi2_3.ffW = sdi2_3.ffW - pmts.alpha * sdi2_3.dffW;
    sdi2_3.ffb = sdi2_3.ffb - pmts.alpha * sdi2_3.dffb;
end
