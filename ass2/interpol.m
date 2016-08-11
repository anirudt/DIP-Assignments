% Interpolation techniques in MATLAB
function y = interpol()
% Main function for testing out different kinds of interpolation techniques
img = imread('../imgs/cameraman.tif');
y = NNinterpol(img, 512, 512);
%imshow(y);
end

function interpolated = NNinterpol(img, M, N)
  [R, C] = size(img);
  j = floor(M/R);
  k = floor(N/C);
  repeat = ones(j, k);
  interpolated = kron(im2double(img), repeat);
end


function interpolated = BLinterpol(img, M, N)

end
