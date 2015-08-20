function [] = fn_SuperpixelMori1(imagename, imageext, N_ev, N_sp3, N_sp2, N_sp1)

% sp_demo.m
%
% See instructions in README.

% % Number of eigenvectors.
% N_ev=5; % layer L-2
% % Number of superpixels coarse/fine.
% N_sp3 = 20; % L-3
% N_sp2=75; % Layer 2
% N_sp1=300; % Layer 1
% imagename = '8068';


% add dir of Ncut code
if ~exist('cncut')
    %     addpath('/cs/fac1/mori/linux/src/matlab-stuff/superpixels/yu_imncut');
    addpath('/home/student1/MATLABcodes/superpixels/yu_imncut');
end

% % add dir of the image
% addpath('/home/student1/MATLABcodes/BSR/BSDS500/data/images/test');


% make a directory
[s,mess,messid] = mkdir(['./',imagename]);
disp(mess);


% I = im2double(imread('img_000070.jpg'));
% I = im2double(imread([imagename,'.jpg']));
I = im2double(imread(['./',imagename, '/',imagename,imageext]));

N = size(I,1);
M = size(I,2);

img_label_hier = cell(4,3);

% ncut parameters for superpixel computation
diag_length = sqrt(N*N + M*M);
par = imncut_sp;
par.int=0;
par.pb_ic=1;
par.sig_pb_ic=0.05;
par.sig_p=ceil(diag_length/50);
par.verbose=0;
par.nb_r=ceil(diag_length/60);
par.rep = -0.005;  % stability?  or proximity?
par.sample_rate=0.2;
par.nv = N_ev;
par.sp = N_sp3;

% Intervening contour using mfm-pb
fprintf('running PB\n');
[emag,ephase] = pbWrapper(I,par.pb_timing);
emag = pbThicken(emag);
par.pb_emag = emag;
par.pb_ephase = ephase;
clear emag ephase;

st=clock;
fprintf('Ncutting...');
[Sp_3,Sp_ev] = imncut_sp(I,par);
fprintf(' took %.2f minutes\n',etime(clock,st)/60);


par.nv = N_sp2;
par.sp = N_sp1;

st=clock;
fprintf('Ncutting...');
[Sp_1,Sp_2] = imncut_sp(I,par);
fprintf(' took %.2f minutes\n',etime(clock,st)/60);

% ################## WARNING ########################
% This code can occasionally produces indices which are out of the order,
% for instance it can produce 1, 2, 3, 5, 8 instead of 1, 2, 3, 4, 5.
% Therefore, we may need to fix this order later.
% ###################################################


I_Sp_ev = segImage(I,Sp_ev);
I_Sp_3 = segImage(I,Sp_3);
I_Sp_2 = segImage(I,Sp_2);
I_Sp_1 = segImage(I,Sp_1);

figure;
subplot(2,2,1); imagesc(Sp_ev); daspect([1 1 1]); axis off; title(['Sp_ev with ',num2str(max(Sp_ev(:),[],1))]);
subplot(2,2,2); imagesc(Sp_3); daspect([1 1 1]); axis off; title(['Sp_3 with ',num2str(max(Sp_3(:),[],1))]);
subplot(2,2,3); imagesc(Sp_2); daspect([1 1 1]); axis off; title(['Sp_2 with ',num2str(max(Sp_2(:),[],1))]);
subplot(2,2,4); imagesc(Sp_1); daspect([1 1 1]); axis off; title(['Sp_1 with ',num2str(max(Sp_1(:),[],1))]);
print('-djpeg','-r100',[imagename,'_segmentation_result.jpg']);
movefile([imagename,'_segmentation_result.jpg'], ['./',imagename]);

figure;
subplot(2,2,1); imagesc(I_Sp_ev); daspect([1 1 1]); axis off; title(['Sp_ev with ',num2str(max(Sp_ev(:),[],1))]);
subplot(2,2,2); imagesc(I_Sp_3); daspect([1 1 1]); axis off; title(['Sp_3 with ',num2str(max(Sp_3(:),[],1))]);
subplot(2,2,3); imagesc(I_Sp_2); daspect([1 1 1]); axis off; title(['Sp_2 with ',num2str(max(Sp_2(:),[],1))]);
subplot(2,2,4); imagesc(I_Sp_1); daspect([1 1 1]); axis off; title(['Sp_1 with ',num2str(max(Sp_1(:),[],1))]);
print('-djpeg','-r100',[imagename,'_superpixel_result.jpg']);
movefile([imagename,'_superpixel_result.jpg'], ['./',imagename]);

figure; imagesc(I_Sp_1); daspect([1 1 1]); axis off; title(['Sp_1 with ',num2str(max(Sp_1(:),[],1))]);
print('-djpeg','-r100',[imagename,'_H1_superpixel_result.jpg']);
movefile([imagename,'_H1_superpixel_result.jpg'], ['./',imagename]);

% ----- store the information ------
% --- update count box ---
img_label_hier{1,1} = N_sp1;
img_label_hier{1,2} = length(unique(Sp_1));
img_label_hier{1,3} = Sp_1;

img_label_hier{2,1} = N_sp2;
img_label_hier{2,2} = length(unique(Sp_2));
img_label_hier{2,3} = Sp_2;

img_label_hier{3,1} = N_sp3;
img_label_hier{3,2} = length(unique(Sp_3));
img_label_hier{3,3} = Sp_3;

img_label_hier{4,1} = N_ev;
img_label_hier{4,2} = length(unique(Sp_ev));
img_label_hier{4,3} = Sp_ev;

save([imagename,'_img_label_hier'], 'img_label_hier');
% move the output to the corresponding folder
movefile([imagename,'_img_label_hier.mat'], ['./',imagename]);
disp(['The results (hierarchical segmentation) are saved in the folder /',imagename]);