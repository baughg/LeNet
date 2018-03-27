function Q = scale_and_quantise( a )
max_a = max(abs(a(:)));
% max_a = (128*128);
% 
% scale_factor = 127.0 / max_a;
% 
% a = a * scale_factor;
a = int32(a);

a = bitshift(a,-2) - 1;
a = bitshift(a,-5);
a(a < 0) = 0;
Q = int8(a);
Q = double(Q);
end

