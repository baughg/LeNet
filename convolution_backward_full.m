function layer1 = convolution_backward_full( input_image_all, weight0_1 )
[H,W,Ci, Co] = size(input_image_all);

for ws = 1:Co
    input_image = input_image_all(:,:,:,ws);
    [H,W,C] = size(input_image);
    [Kh,Kw,Fo,Fi] = size(weight0_1);
    
    
    row_pad = (Kh - 1) / 2;
    col_pad = (Kw - 1) / 2;
    
    layer1 = zeros(H,W,Fo);
    
    
    input_image_pad = zeros(H+2*row_pad,W+2*col_pad);
    
    for c = 1:C
        input_image_pad((row_pad+1):(end-row_pad),(col_pad+1):(end-col_pad)) = input_image(:,:,c);
        
        for f = 1:Fo
            kernel = weight0_1(:,:,f,c);
            kernel = rot90(squeeze(kernel),2);
            conv_out = conv2(input_image_pad,kernel,'valid');
            layer1(:,:,f) =  layer1(:,:,f) + conv_out;
        end
    end
    
    
    layer1 = squeeze(layer1);
end
end

