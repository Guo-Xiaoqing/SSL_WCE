radius = 282.5;
[X,Y] = meshgrid([-radius:1:radius]/radius,[-radius:1:radius]/radius);
X = X(((radius+0.5)*2-512)/2+1:radius+256.5,((radius+0.5)*2-512)/2+1:radius+256.5);
Y = Y(((radius+0.5)*2-512)/2+1:radius+256.5,((radius+0.5)*2-512)/2+1:radius+256.5);
%[X,Y] = meshgrid([-255.5:1:255.5]/255.5,[-255.5:1:255.5]/255.5);
U = X .* sqrt(1-Y.^2/2.0);
V = Y .* sqrt(1-X.^2/2.0);
flow = zeros([512,512,2]);
flow(:,:,1) = (-X+U)*radius;
flow(:,:,2) = (-Y+V)*radius;

folder = 'vas3/';%inf2,inf2_seg,nor1,vas3,vas3_seg,wcetest
files = dir(fullfile('./WCE/test/',folder,'*.jpg'));
for i = 1:1:length(files)
    file_name = files(i).name;
    img = imread(['./WCE/test/',folder,file_name]);
    warped = imwarp(img,flow);
    %imshow(warped)
    imwrite(warped,['./WCE/warped_test/',folder,file_name])
end
