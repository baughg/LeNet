function [layer1, delta_weight] = convolution_backward_full( input_image_all, weight0_1, layer1_fp )
[H,W,Ci, Co] = size(input_image_all);
layer1 = zeros(H,W,Co);
delta_weight = zeros(size(weight0_1));

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
        layer1(:,:,ws) =  layer1(:,:,ws) + conv_out;
    end
    
    
    
end

layer1 = squeeze(layer1);
layer1(layer1_fp <= 0) = 0;
end

