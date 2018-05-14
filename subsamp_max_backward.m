function [ layer0_error ] = subsamp_max_backward( layer0,  layer1_error)
% features->layer3, errors->layer3, errors->layer4
[H,W,C] = size(layer0);
layer0_error = zeros(size(layer0));

for c = 1:C
    for y = 1:2:H
        yy = round(y / 2);
        for x = 1:2:W
            B = layer0(y:y+1,x:x+1);
            M = B == max(B(:));          
            xx = round(x / 2);
            val = layer1_error(yy,xx,c);
            B = B * 0;
            B(M) = val;
            layer0_error(y:y+1,x:x+1,c) = B;
        end
    end    
end
end

