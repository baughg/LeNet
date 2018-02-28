function Wght = read_array(fid,W,H,Cn, Cp )

Wght = zeros(W,H,Cn,Cp);

for co = 1:Cp
    weight = zeros(H,W,Cn);
    for ci = 1:Cn
        weight(:,:,ci) = fread(fid,[H W],'float64')';
    end
    
    Wght(:,:,:,co) = weight;
end

Wght = squeeze(Wght);
end

