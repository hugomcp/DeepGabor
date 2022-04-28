function ret=position_2_input_region(pos, kernel_size, img_size, resize_factor)
% returns [row_begin, row_end, col_begin, col_end]

img_size_c = [img_size(1), img_size(2)]-kernel_size+1;

img_size_r = ceil(img_size_c.*resize_factor);


row_r = mod(pos, img_size_r(1));
if row_r == 0 
    row_r = img_size_r(1);
end

col_r = ceil(pos/img_size_r(1));


row_c = img_size_c(1)*row_r/img_size_r(1);
col_c = img_size_c(2)*col_r/img_size_r(2);

row_i=row_c+(kernel_size-1)/2;
col_i=col_c+(kernel_size-1)/2;

ret=round([row_i-(kernel_size-1)/2, row_i+(kernel_size-1)/2, col_i-(kernel_size-1)/2, col_i+(kernel_size-1)/2]);

