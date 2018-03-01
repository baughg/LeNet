% typedef struct LeNet5
% {
% 	double weight0_1[1][6][5][5];
% 	double weight2_3[6][16][5][5];
% 	double weight4_5[16][120][5][5];
% 	double weight5_6[120][10];
% 
% 	double bias0_1[6];
% 	double bias2_3[16];
% 	double bias4_5[120];
% 	double bias5_6[10];
% 
% }LeNet5;
fid = fopen('input_image.bin','rb');
input_image = fread(fid,[28 28],'uint8')';
fclose(fid);

input_image = uint8(input_image);

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

fid = fopen('layer1.bin','rb');
layer1_ref = read_array(fid,28,28,6,1);
fclose(fid);

% max pool layer 1
layer2 = max_pool( layer1 );

fid = fopen('layer2.bin','rb');
layer2_ref = read_array(fid,14,14,6,1);
fclose(fid);

% layer 2 convolution
layer3 = convolution_relu( layer2, weight2_3, bias2_3, 0 );

fid = fopen('layer3.bin','rb');
layer3_ref = read_array(fid,10,10,16,1);
fclose(fid);

% max pool layer 3
layer4 = max_pool( layer3 );

fid = fopen('layer4.bin','rb');
layer4_ref = read_array(fid,5,5,16,1);
fclose(fid);

% layer 4 convolution
layer5 = convolution_relu( layer4, weight4_5, bias4_5, 0 );

fid = fopen('layer5.bin','rb');
layer5_ref = read_array(fid,1,1,120,1);
fclose(fid);

output = sum(repmat(layer5',10,1) .* weight5_6, 2);
output = output + bias5_6;
output = output .* double(output > 0); % relu;
output = output ./ sum(output);

