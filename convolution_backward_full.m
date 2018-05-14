function [layerX, delta_weight, deltas_bias] = convolution_backward_full( input_image_all, weight0_1, layerX_fp )
[H,W,Ci, Co] = size(input_image_all);
layerX = zeros(H,W,Co);
delta_weight = zeros(size(input_image_all));


for ws = 1:Co
    input_image = input_image_all(:,:,:,ws);
    [H,W,C] = size(input_image);
    [Kh,Kw,Fo,Fi] = size(weight0_1);
    
    row_pad = (Kh - 1) / 2;
    col_pad = (Kw - 1) / 2;
    
    input_image_pad = zeros(H+2*row_pad,W+2*col_pad);
    
    for c = 1:C
        input_image_pad((row_pad+1):(end-row_pad),(col_pad+1):(end-col_pad)) = input_image(:,:,c);
        kernel = weight0_1(:,:,c,1);
        kernel = rot90(squeeze(kernel),2);
        conv_out = conv2(input_image_pad,kernel,'valid');
        layerX(:,:,ws) =  layerX(:,:,ws) + conv_out;
    end
end

layerX = squeeze(layerX);
layerX(layerX_fp <= 0) = 0;

[H,W,Ci, Co] = size(layerX_fp);
[Kh,Kw,Fo,Fi] = size(weight0_1);

deltas_bias = zeros(Fo,1);

for w = 1:Fo
    kernel = weight0_1(:,:,w,1);
    deltas_bias(w) = sum(kernel(:));
end

for c = 1:Ci
    input_image_pad = zeros(H+2*row_pad,W+2*col_pad);
    input_image_pad((row_pad+1):(end-row_pad),(col_pad+1):(end-col_pad)) = layerX_fp(:,:,c);
    
    for w = 1:Fo
        kernel = weight0_1(:,:,w,1);
        
        kernel = rot90(squeeze(kernel),2);
        conv_out = conv2(input_image_pad,kernel,'valid');
        delta_weight(:,:,w,c) = conv_out;
    end
end
end

