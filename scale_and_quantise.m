function Q = scale_and_quantise( a )

scale_factor = 127.0 / max(abs(a(:)));

a = a * scale_factor;
Q = int8(a);
Q = double(Q);
end

