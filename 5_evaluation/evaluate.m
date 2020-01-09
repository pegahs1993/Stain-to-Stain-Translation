%%
path_image = '\Ground_truth\';
path_fake = '\stst\';

n = 500;
R_mse = zeros(1,n);
R_psnr = zeros(1,n);
R_ssim = zeros(1,n);
R_msssim = zeros(1,n);
R_FSIM = zeros(1,n);

for i=1:500
    img = sprintf('%d.tiff',i);
    fake = sprintf('%d.tiff',i);
    
    image_files = fullfile(path_image, img);
    fake_files = fullfile(path_fake, fake);
    

    img_A = imread(image_files);
    fake_A=  imread(fake_files);
    
    R_mse(1,i) = immse(img_A,fake_A);
    R_psnr(1,i)= psnr(img_A,fake_A);
    R_ssim(1,i)= ssim(img_A,fake_A);
    R_msssim(1,i) = msssim(img_A,fake_A);
    R_FSIM(1,i) = FeatureSIM(img_A,fake_A);
  
    X = ['step = ',num2str(i)];
    disp(X)
end
[omse,imse]=sort(R_mse);
[opsnr,ipsnr]=sort(R_psnr);
[ossim,issim]=sort(R_ssim);
[omsssim,imsssim]=sort(R_msssim);
[ofsim,ifsim]=sort(R_FSIM);

%% mean & SD
mi_ssim = mean(R_ssim);
stdv_ssim = std(R_ssim);

mi_psnr = mean(R_psnr);
stdv_psnr = std(R_psnr);

mi_mse = mean(R_mse);
stdv_mse = std(R_mse);

mi_msssim = mean(R_msssim);
stdv_msssim = std(R_msssim);

mi_fsim = mean(R_FSIM);
stdv_fsim = std(R_FSIM);


%% save section
save('\r_sp.mat');
%% save excel
path = '\r_sp.xlsx';
xlswrite(path,R_ssim',1,'A1')
xlswrite(path,ossim',1,'B1')
xlswrite(path,issim',1,'C1')
xlswrite(path,mi_ssim',1,'D1')
xlswrite(path,stdv_ssim',1,'E1')

xlswrite(path,R_mse',1,'F1')
xlswrite(path,omse',1,'G1')
xlswrite(path,imse',1,'H1')
xlswrite(path,mi_mse',1,'I1')
xlswrite(path,stdv_mse',1,'J1')

xlswrite(path,R_psnr',1,'K1')
xlswrite(path,opsnr',1,'L1')
xlswrite(path,ipsnr',1,'M1')
xlswrite(path,mi_psnr',1,'N1')
xlswrite(path,stdv_psnr',1,'O1')

xlswrite(path,R_msssim',1,'P1')
xlswrite(path,omsssim',1,'Q1')
xlswrite(path,imsssim',1,'R1')
xlswrite(path,mi_msssim',1,'S1')
xlswrite(path,stdv_msssim',1,'T1')

xlswrite(path,R_FSIM',1,'U1')
xlswrite(path,ofsim',1,'V1')
xlswrite(path,ifsim',1,'W1')
xlswrite(path,mi_fsim',1,'X1')
xlswrite(path,stdv_fsim',1,'Y1')


