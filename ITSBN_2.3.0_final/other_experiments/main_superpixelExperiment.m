clear all;
close all;
clc;

addpath('../');

imagename = '1_23_s', imageext = '.bmp'

% ====================== Mori's NCUT ==================================
    % Since we have L = 6 levels for an image, there will be 4 intermediate
    % levels, 4 to 1. There are several settings the user can pick from.
    % =====================================================================
%     N_ev = 5; N_sp3 = 20; N_sp2 = 75; N_sp1 = 300; % setting#1 takes 230 sec/itt
%     N_ev = 5; N_sp3 = 20; N_sp2 = 50; N_sp1 = 200; % setting#2 46 sec/itt
%     N_ev = 5; N_sp3 = 20; N_sp2 = 50; N_sp1 = 150; % setting#3 15 sec/itt
%     N_ev = 5; N_sp3 = 20; N_sp2 = 50; N_sp1 = 100; % setting#4 5 sec/itt
%     N_ev = 5; N_sp3 = 20; N_sp2 = 30; N_sp1 = 50; % setting#5 2 sec/itt
% N_ev = 5; N_sp3 = 20; N_sp2 = 80; N_sp1 = 250; % setting#5 2 sec/itt
N_ev = 5; N_sp3 = 20; N_sp2 = 60; N_sp1 = 150; % setting#5 2 sec/itt
    fn_SuperpixelMori2(imagename, imageext, N_ev, N_sp3, N_sp2, N_sp1);