%-----------------------------------------------------------
% Preamble
%-----------------------------------------------------------
close all;
clear all;
clc;
format('compact');
graphics_toolkit gnuplot;
addpath('funcs');
addpath(genpath('../sim_edu_bbt/funcs/'));
addpath(genpath('../lib/'));
%-----------------------------------------------------------


%-----------------------------------------------------------
% Parameters
%-----------------------------------------------------------
source('../sim_edu_bbt/system_parameters.m');
fsymb = 1./Tsymb;
fs    = 1./Ts;
%-----------------------------------------------------------


%-----------------------------------------------------------
% Design of Pre-Filter
%-----------------------------------------------------------
N        = 1;
f_center = fsymb/2;
bw       = fsymb/10; % TODO: esto fue a ojo
[pre_b, pre_a]   = bp_synth(N, f_center, bw, fs);
%-----------------------------------------------------------


%-----------------------------------------------------------
% Design of BandPass Filter
%-----------------------------------------------------------
N            = 1;
f_center     = fsymb;
bw           = fsymb/40; % TODO: esto fue a ojo
[bp_b, bp_a] = bp_synth(N, f_center, bw, fs);
%-----------------------------------------------------------


%-----------------------------------------------------------
% Plot filter response
%-----------------------------------------------------------
figure(1);
[H, W] = freqz(pre_b, pre_a, 128, fs);
freqz_plot(W, H);
figure(2);
[H, W] = freqz(bp_b, bp_a, 128, fs);
freqz_plot(W, H);
%-----------------------------------------------------------


%-----------------------------------------------------------
% Write filter data
%-----------------------------------------------------------
data = [pre_b;pre_a];
dlmwrite('./data/symb_sync_pre_filter.dat', data);
data = [bp_b;bp_a];
dlmwrite('./data/symb_sync_bp_filter.dat', data);
%-----------------------------------------------------------
