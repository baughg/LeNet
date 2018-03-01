input_image = imread('test_data/seven_line.png');

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


% layer 1 convolution
layer1 = convolution_relu( input_image, weight0_1, bias0_1, 1 );
% max pool layer 1
layer2 = max_pool( layer1 );
% layer 2 convolution
layer3 = convolution_relu( layer2, weight2_3, bias2_3, 0 );
% max pool layer 3
layer4 = max_pool( layer3 );
% layer 4 convolution
layer5 = convolution_relu( layer4, weight4_5, bias4_5, 0 );

output = sum(repmat(layer5',10,1) .* weight5_6, 2);
output = output + bias5_6;
output = output .* double(output > 0); % relu;


output = output ./ sum(output);

digit = find(output == max(output)) - 1;



