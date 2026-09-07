function Quantum_Antenna()

clear; close all; clc;

%% 1. Paths 
scriptDir = fileparts(mfilename('fullpath'));
dataDir   = fullfile(scriptDir, '..', 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end

%% 2. Physical parameters

f0   = 3e12;                 % transition frequency [Hz]  (lambda = 100 um)
w0   = 2*pi*f0;              % transition angular frequency [rad/s]
G    = 2*pi*20e9;            % total decay rate [rad/s] (T1 ~ 8 ps, intraband)
e    = 1.602176634e-19;      % elementary charge [C]
z12  = 2e-9;                 % dipole matrix element length [m]
dip  = e*z12;                % dipole matrix element d = e*z12 [C.m]
eps0 = 8.8541878128e-12;     % vacuum permittivity [F/m]
c    = 2.99792458e8;         % speed of light [m/s]
hbar = 1.054571817e-34;      % reduced Planck constant [J.s]
Dl   = 0;                    % detuning (resonant drive) [rad/s]

fprintf('== Quantum Antenna Element Simulation ==\n');
fprintf('f0 = %.2f THz, G/2pi = %.1f GHz, d = %.3e C.m\n', ...
        f0/1e12, G/(2*pi)/1e9, dip);

%% 3. Optical Bloch equations
%  x = [u; v; w],  coherence rho_eg = (u + i*v)/2,  inversion w = rho_ee-rho_gg
%    u' = -G/2*u + Dl*v
%    v' = -Dl*u - G/2*v + Om*w
%    w' = -Om*v - G*(w+1)
A_of = @(Om) [ -G/2,  Dl ,   0 ;
               -Dl , -G/2,  Om ;
                 0 ,  -Om,  -G ];
b_vec = [0; 0; G];                      % w' = -Om*v - G*w - G  ->  A*x = b
bloch_rhs = @(t, x, Om) A_of(Om)*x - [0;0;G];   % w' = -Om*v - G*w - G

%  fixed-step RK4 integrator (no toolbox required)
rk4 = @(rhs, x0, tgrid) rk4_integrate(rhs, x0, tgrid);   % local function below

%% ---------------- (a) Rabi oscillations: weak vs strong drive ------------
t_ps   = (0:0.02:120)';                 % time grid [ps]
t_s    = t_ps*1e-12;
Om_w   = 1*G;                           % weak drive   Om/2pi =  20 GHz
Om_s   = 4*G;                           % strong drive Om/2pi =  80 GHz
x0     = [0; 0; -1];                    % QD starts in ground state

Xw = rk4(@(t,x) bloch_rhs(t,x,Om_w), x0, t_s);
Xs = rk4(@(t,x) bloch_rhs(t,x,Om_s), x0, t_s);
rho_ee_w = (1 + Xw(:,3))/2;
rho_ee_s = (1 + Xs(:,3))/2;

fid = fopen(fullfile(dataDir,'rabi.csv'),'w');
fprintf(fid,'t_ps,rho_ee_Om1G,rho_ee_Om4G\n');
fprintf(fid,'%.4f,%.6f,%.6f\n',[t_ps, rho_ee_w, rho_ee_s]');
fclose(fid);

%% ---------------- (b) Radiated power vs drive strength -------------------
Om_ratio = (0.01:0.02:10)'; Om_ratio(end) = 10;   % exact endpoint             % Om/G sweep
n        = numel(Om_ratio);
coh      = zeros(n,1);
for k = 1:n
    Om  = Om_ratio(k)*G;
    xss = A_of(Om)\b_vec;               % steady state
    coh(k) = sqrt(xss(1)^2 + xss(2)^2)/2;   % |<s->|_ss
end
Pq_W = w0^4 .* (2*dip*coh).^2        / (12*pi*eps0*c^3);   % quantum [W]
Pc_W = w0^4 .* (2*dip*(Om_ratio)).^2 / (12*pi*eps0*c^3);   % classical [W]
Pq   = Pq_W*1e18;  Pc = Pc_W*1e18;                          % -> aW

% analytic optimum:  dPq/dOm = 0  ->  Om_pk = G/sqrt(2), coh_max = 1/(2*sqrt2)
Om_pk  = G/sqrt(2);
coh_max= Om_pk*G/(2*Om_pk^2 + G^2);
Pq_max = w0^4*(2*dip*coh_max)^2/(12*pi*eps0*c^3)*1e18;

fid = fopen(fullfile(dataDir,'power_sweep.csv'),'w');
fprintf(fid,'Om_over_G,P_quant_aW,P_class_aW\n');
fprintf(fid,'%.3f,%.6e,%.6e\n',[Om_ratio, Pq, Pc]');
fclose(fid);

%% ---------------- (c) Emission spectrum (QRT -> Mollow) ------------------
tau   = (0 : 1/50 : 60)'/G;             % delay grid [s], up to 60/G
nT    = numel(tau);
f_ax  = linspace(-260e9, 260e9, 801);   % spectrum axis (rot. frame) [Hz]
Om_set = [1, 5]*G;                      % weak & strong drive
S_norm = zeros(numel(f_ax), numel(Om_set));
for m = 1:numel(Om_set)
    Om  = Om_set(m);
    xss = A_of(Om)\b_vec;               % steady state
    rho_ee_ss = (1 + xss(3))/2;
    rho_eg_ss = (xss(1) + 1i*xss(2))/2;
    % QRT initial condition for g(tau) = [<x s->, <y s->, <z s->]
    g0 = [ rho_ee_ss ; -1i*rho_ee_ss ; -rho_eg_ss ];
    % homogeneous Bloch propagation (regression theorem)
    Gh = rk4(@(t,x) A_of(Om)*x, g0, tau);
    C  = 0.5*(Gh(:,1) + 1i*Gh(:,2));            % <s+(tau) s-(0)>
    S  = real( (exp(1i*2*pi*(f_ax(:)*tau(:)')) * C) * (tau(2)-tau(1)) )/pi;
    S_norm(:,m) = S/max(S);
    intS = trapz(f_ax, S);                      % check: ~= rho_ee
    fprintf('Spectrum check: integral(S)df = %.4f vs rho_ee/(2pi) = %.4f (Om=%.1f G)\n', ...
            intS, rho_ee_ss/(2*pi), Om/G);
end

fid = fopen(fullfile(dataDir,'spectrum.csv'),'w');
fprintf(fid,'f_GHz,S_Om1G,S_Om5G\n');
fprintf(fid,'%.3f,%.6f,%.6f\n',[f_ax(:)/1e9, S_norm]');
fclose(fid);

%% ---------------- (d) Far-field pattern vs drive -------------------------
th      = (0:1:360)';                   % polar angle [deg]
Om_p    = [1/sqrt(2), 3, 8]*G;          % near-optimal, moderate, strong
scl     = zeros(1,3);
for m = 1:3
    Om  = Om_p(m);
    xss = A_of(Om)\b_vec;
    coh_m   = sqrt(xss(1)^2 + xss(2)^2)/2;
    scl(m)  = (coh_m/coh_max)^2;        % quantum scaling of pattern
end
pat = sin(th*pi/180).^2 .* scl;         % sin^2(theta) * |<s->|^2 / max

fid = fopen(fullfile(dataDir,'pattern.csv'),'w');
fprintf(fid,'theta_deg,P_Om0p71G,P_Om3G,P_Om8G\n');
fprintf(fid,'%.1f,%.6f,%.6f,%.6f\n',[th, pat]');
fclose(fid);

%% 4.  Summary numbers
Pq_at_10G = interp1(Om_ratio, Pq, 10, 'linear');
Pc_at_10G = interp1(Om_ratio, Pc, 10, 'linear');
fid = fopen(fullfile(dataDir,'summary.txt'),'w');
fprintf(fid,'Peak coherent radiated power P_max = %.4f aW at Om = G/sqrt2 (Om/2pi = %.2f GHz)\n', ...
        Pq_max, Om_pk/2/pi/1e9);
fprintf(fid,'Quantum power at Om = 10G : %.4e aW\n', Pq_at_10G);
fprintf(fid,'Classical power at Om = 10G: %.4f aW (overestimate x %.0f)\n', ...
        Pc_at_10G, Pc_at_10G/Pq_at_10G);
fprintf(fid,'Pattern scale factors (Om = 0.71G / 3G / 8G): %.3f / %.3f / %.3f\n', scl);
fprintf(fid,'Radiative rate from d: G_rad/2pi = %.2f Hz -> antenna coupling is essential\n', ...
        (w0^3*dip^2/(3*pi*eps0*hbar*c^3))/2/pi);
fclose(fid);
type(fullfile(dataDir,'summary.txt'));

%% 5. Figures 
MAKE_FIGS = true;
if MAKE_FIGS
  try
    figDir = fullfile(scriptDir, '..', 'figures');
    if ~exist(figDir,'dir'), mkdir(figDir); end
    % (a) Rabi
    figure('Color','w'); plot(t_ps, rho_ee_w, 'LineWidth', 2); hold on;
    plot(t_ps, rho_ee_s, 'LineWidth', 2);
    xlabel('Time (ps)'); ylabel('\rho_{ee}'); legend('Om = G','Om = 4G');
    grid on; title('(a) Rabi oscillations of the QD emitter');
    print(gcf, fullfile(figDir,'fig_a_rabi.png'), '-dpng', '-r200');
    
    % (b) Power vs drive
    figure('Color','w'); plot(Om_ratio, Pq, 'LineWidth', 2.5); hold on;
    plot(Om_ratio, Pc, '--', 'LineWidth', 2);
    xlabel('Drive strength  \Omega/\Gamma'); ylabel('Radiated power (aW)');
    legend('Quantum (this work)','Classical dipole','Location','northwest');
    grid on; title('(b) Radiated power vs drive strength');
    print(gcf, fullfile(figDir,'fig_b_power.png'), '-dpng', '-r200');
    
    % (c) Spectrum
    figure('Color','w'); plot(f_ax/1e9, S_norm(:,1), 'LineWidth', 2); hold on;
    plot(f_ax/1e9, S_norm(:,2), 'LineWidth', 2);
    xlabel('Frequency detuning (GHz)'); ylabel('S(\omega) (norm.)');
    legend('\Omega = \Gamma','\Omega = 5\Gamma'); grid on;
    title('(c) Emission spectrum (Mollow triplet)');
    print(gcf, fullfile(figDir,'fig_c_spectrum.png'), '-dpng', '-r200');
    
    % (d) Pattern
    figure('Color','w'); polarplot(th*pi/180, pat(:,1), 'LineWidth', 2); hold on;
    polarplot(th*pi/180, pat(:,2), 'LineWidth', 2);
    polarplot(th*pi/180, pat(:,3), 'LineWidth', 2);
    title('(d) Far-field pattern vs drive');
    legend('\Omega = 0.71\Gamma','\Omega = 3\Gamma','\Omega = 8\Gamma');
    print(gcf, fullfile(figDir,'fig_d_pattern.png'), '-dpng', '-r200');
    disp('Figures written to ../figures');
    
catch err
    fprintf(['Graphics unavailable in this environment (%s).\n' ...
             'Data exported to ../data - run this file in MATLAB to plot.\n'], ...
             err.message);
  end
end

fprintf('DONE\n');
end

function X = rk4_integrate(rhs, x0, tgrid)
%RK4_INTEGRATE  Fixed-step classical Runge-Kutta 4 integrator.
%   X = rk4_integrate(rhs, x0, tgrid) integrates dx/dt = rhs(t,x) over the
%   uniform time grid tgrid starting from x0. Returns X with one state
%   vector per row. No toolboxes required (MATLAB / GNU Octave compatible).
    n  = numel(tgrid);
    dt = tgrid(2) - tgrid(1);
    X  = zeros(n, numel(x0));
    x  = x0(:);
    X(1,:) = x(:).';
    for k = 1:n-1
        k1 = rhs(tgrid(k),      x);
        k2 = rhs(tgrid(k)+dt/2, x + dt/2*k1);
        k3 = rhs(tgrid(k)+dt/2, x + dt/2*k2);
        k4 = rhs(tgrid(k)+dt,   x + dt*k3);
        x  = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
        X(k+1,:) = x(:).';
    end
end
