% Clear all previous data
clc, clear all, close all;
%% Display results of each method
verbose = 1;

%% Color Deconvolution using Our Implementation with Standard Stain Matrix
path_image = '\stst_patches\';

path_H = '\stst\H\';
path_E = '\stst\E\';
path_Bg = '\stst\Bg\';


for i=1:500
    img = sprintf('%d.tiff',i); 
    image_files = fullfile(path_image, img);
    NormImage = imread(image_files);
    % Get pseudo-coloured deconvolved channels
    stains = Deconvolve( NormImage, [], 0 );
    [H, E, Bg] = PseudoColourStains( stains, [] );
    % % save image
    image_H = fullfile(path_H, img);
    imwrite(H,image_H)
    image_E = fullfile(path_E, img);
    imwrite(E,image_E)
    image_Bg = fullfile(path_Bg, img);
    imwrite(Bg,image_Bg)
end
