function [ rating ] = msssim ( f, g )
    % Copyright(c) Christophe Charrier <christophe.charrier@unicaen.fr>, 2010
    
    K = [0.01 0.03];
    winsize = 11;
    sigma = 1.5;
    window = fspecial('gaussian', winsize, sigma);
    level = 5;
    weights = [0.0448 0.2856 0.3001 0.2363 0.1333];
    method = 'product';
    
    [rating, ld, cd, sd]= ssim_mscale_new_CC(rgb2gray(f), rgb2gray(g), K, window, level, weights, method);
%     x = [rating, real(ld), real(cd), real(sd)];
    
end

function [overall_mssim l_ssim c_ssim cs_ssim] = ...
    ssim_mscale_new_CC(img1, img2, K, window, level, weight, method)
    
    % Multi-scale Structural Similarity Index (MS-SSIM)
    % Z. Wang, E. P. Simoncelli and A. C. Bovik,
    % "Multi-scale structural similarity for image quality assessment,"
    % Invited Paper, IEEE Asilomar Conference on Signals, Systems and Computers,
    % Nov. 2003
    
    if (nargin < 2) || (nargin > 7),
        overall_mssim = -Inf;
        return;
    end
    
    if ~exist('K', 'var'),
        K = [0.01 0.03];
    end

    if ~exist('window', 'var'),
        window = fspecial('gaussian', 11, 1.5);
    end
    
    if ~exist('level', 'var'),
        level = 5;
    end
    
    if ~exist('weight', 'var'),
        weight = [0.0448 0.2856 0.3001 0.2363 0.1333];
    end
    
    if ~exist('method', 'var'),
        method = 'product';
    end

    if size(img1) ~= size(img2),
        overall_mssim = -Inf;
        return;
    end
    
    [M N] = size(img1);
    if (M < 11) || (N < 11),
        overall_mssim = -Inf;
        return;
    end
    
    if (length(K) ~= 2)
        overall_mssim = -Inf;
        return;
    end

    if (K(1) < 0) || (K(2) < 0),
        overall_mssim = -Inf;
        return;
    end
    
    [H W] = size(window);
    
    if ((H*W)<4) || (H>M) || (W>N),
        overall_mssim = -Inf;
        return;
    end
    
    if level < 1,
        overall_mssim = -Inf;
        return;
    end
    
    
    min_img_width = min(M, N)/(2^(level-1));
    max_win_width = max(H, W);
    if min_img_width < max_win_width,
        overall_mssim = -Inf;
        return;
    end
    
    if (length(weight) ~= level) || (sum(weight) == 0),
        overall_mssim = -Inf;
        return;
    end
    
    if ~strcmp(method,'wtd_sum') && ~strcmp(method,'product'),
        overall_mssim = -Inf;
        return;
    end
    
    downsample_filter = ones(2)./4;
    im1 = img1;
    im2 = img2;
    for l = 1:level
        [mssim_L_array(l) mssim_C_array(l) mssim_CS_array(l) ...
         ssim_L_map_array{l} ssim_C_map_array{l}  cs_map_array{l}] ...
            = ssim_index_new_CC(im1, im2, K, window);
        [M N] = size(im1);
        filtered_im1 = filter2(downsample_filter, im1, 'valid');
        filtered_im2 = filter2(downsample_filter, im2, 'valid');
        clear im1 im2;
        im1 = filtered_im1(1:2:M-1, 1:2:N-1);
        im2 = filtered_im2(1:2:M-1, 1:2:N-1);
        ds_img_array1{l} = im1;
        ds_img_array2{l} = im2;
    end
    
    if strcmp(method,'product'),
        %overall_mssim = prod(mssim_array.^weight);
        l_ssim = mssim_L_array(1);
        c_ssim = prod(mssim_C_array(1:level-1).^weight(1:level-1));
        cs_ssim = prod(mssim_CS_array(1:level-1).^weight(1:level-1));
        overall_mssim = l_ssim*c_ssim*cs_ssim;
    else
        weight = weight./sum(weight);
        l_ssim = mssim_L_array(1);
        c_ssim = sum(mssim_C_array(1:level-1).*weight(1:level-1));
        cs_ssim = sum(mssim_CS_array(1:level-1).*weight(1:level-1));
        overall_mssim = l_ssim+c_ssim+cs_ssim;
    end
    
end


function [ml, mc , mcs, l_map, c_map , cs_map] = ...
    ssim_index_new_CC(img1, img2, K, window)
    
    if (nargin < 2) || (nargin > 4),
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
    
    if size(img1) ~= size(img2),
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
    
    [M N] = size(img1);
    
        % Default settings.
    if nargin == 2,
        if (M < 11) || (N < 11),
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
        window = fspecial('gaussian', 11, 1.5);
        K(1) = 0.01;
        K(2) = 0.03;
    end
    
    if nargin == 3,
        if (M < 11) || (N < 11),
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
        window = fspecial('gaussian', 11, 1.5);
        if length(K) == 2,
            if (K(1) < 0) || (K(2) < 0),
                ssim_index = -Inf;
                ssim_map = -Inf;
                return;
            end
        else
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    end
    
    if nargin == 4,
        [H W] = size(window);
        if ((H*W) < 4) || (H > M) || (W > N),
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
        if length(K) == 2,
            if (K(1) < 0) || (K(2) < 0),
                ssim_index = -Inf;
                ssim_map = -Inf;
                return;
            end
        else
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    end
    
    C1 = (K(1)*255)^2;
    C2 = (K(2)*255)^2;
    window = window/sum(sum(window));
    
    mu1   = filter2(window, img1, 'valid');
    mu2   = filter2(window, img2, 'valid');
    mu1_sq = mu1.*mu1;
    mu2_sq = mu2.*mu2;
    mu1_mu2 = mu1.*mu2;
    sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
    sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
    sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
    sigma1 = sqrt(sigma1_sq);
    sigma2 = sqrt(sigma2_sq);
    
    l_map = (2*mu1_mu2 + C1)./(mu1_sq + mu2_sq + C1);
    c_map = (2*sigma1.*sigma2 + C2)./(sigma1_sq + sigma2_sq + C2);
    cs_map = (2*sigma12 + C2)./(2*sigma1.*sigma2 + C2);
    
    ml = mean2(l_map);
    mc = mean2(c_map);
    mcs = mean2(cs_map);
    
end