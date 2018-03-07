input_image = imread('test_data/seven.png');

input_image = uint8(rgb2gray(input_image));

figure(1);
image(input_image); axis image; axis off; colormap(gray(256));

% normalise image
input_image = double(input_image);
input_mean = mean(input_image(:));
input_std = std(input_image(:));



input_image = input_image - input_mean;
input_image = input_image ./ input_std;



figure(2);
imagesc(input_image); axis image; axis off;

fid = fopen('model.dat','rb');

weight0_1 = read_array(fid,5,5,6,1);
weight2_3 = read_array(fid,5,5,16,6);
weight4_5 = read_array(fid,5,5,120,16);
weight5_6 = read_array(fid,1,1,10,120);
bias0_1 = fread(fid,6,'float64');
bias2_3 = fread(fid,16,'float64');
bias4_5 = fread(fid,120,'float64');
bias5_6 = fread(fid,10,'float64');
fclose(fid);

scale_factor = 127;

input_image = scale_and_quantise(input_image);

fid = fopen('test_data_8bit/input_i8.bin','wb');
out = int8(input_image');
fwrite(fid,out(:),'int8');
fclose(fid);

weight0_1 = quantise_array(weight0_1,scale_factor);
weight2_3 = quantise_array(weight2_3,scale_factor);
weight4_5 = quantise_array(weight4_5,scale_factor);
weight5_6 = quantise_array(weight5_6,scale_factor);
bias0_1 = quantise_array(bias0_1,scale_factor);
bias2_3 = quantise_array(bias2_3,scale_factor);
bias4_5 = quantise_array(bias4_5,scale_factor);
bias5_6 = quantise_array(bias5_6,scale_factor);

write_array('test_data_8bit/weight0_1_i8.bin',weight0_1 );
write_array('test_data_8bit/weight2_3_i8.bin',weight2_3 );
write_array('test_data_8bit/weight4_5_i8.bin',weight4_5 );
write_array('test_data_8bit/weight5_6_i8.bin',weight5_6 );


% layer 1 convolution
layer1 = convolution_relu( input_image, weight0_1, bias0_1, 1 );

% max pool layer 1
layer2 = max_pool( layer1 );
layer2 = scale_and_quantise(layer2);
% layer 2 convolution
layer3 = convolution_relu( layer2, weight2_3, bias2_3, 0 );

% max pool layer 3
layer4 = max_pool( layer3 );
layer4 = scale_and_quantise(layer4);
% layer 4 convolution
layer5 = convolution_relu( layer4, weight4_5, bias4_5, 0 );
layer5 = scale_and_quantise(layer5);

dot_prod = repmat(layer5',10,1) .* weight5_6;
dot_prod = scale_and_quantise(dot_prod);

output = sum(dot_prod, 2);
output = output + bias5_6;
output = output .* double(output > 0); % relu;


output = output ./ sum(output);

digit = find(output == max(output)) - 1;



