%% Droplet shape
% reconstruct the liquid droplet shape on a tilted plane with
% angle (alpha) and bond number (Bo) numerically using finite-differencing method.
% Akash Kumar
clc; clear all; close all;

%% Global variables
range_phi = pi;                            % range of phi values [0,180]
range_w   = 1;                             % range of w values [0,1]
n_phi     = 23;                            % no. of nodes for phi
n_w       = 1001;                          % no. of nodes for W
Bo        = 2;                             % bond number value
alpha     = pi/3;                          % tilt angle (radians)
del_phi   = range_phi/(n_phi - 1);         % step-size for phi
del_w     = range_w/(n_w - 1);             % step-size for W
W         = 0:del_w:range_w;               % dimensionless W values
phi       = 0:del_phi:range_phi;           % phi values (radians)
g         = 9.8;                           % acc. due to gravity (m/s^2)
rho_air   = 1.225;                         % air density (Kg/m^3)
rho_water = 1000;                          % water density (Kg/m^3)
del_rho   = rho_water - rho_air;           % change in density (Kg/m^3)
sigma     = 0.07286;                       % surface tension of water (N/m)
b         = sqrt((sigma*Bo)/(del_rho*g));  % principal radii at origin (m)
epsilon   = 1e-4;                          % for limiting values
clip_idx  = n_w;                           % value for clipping

%% Function initialization
Q = zeros(n_phi,n_w);                      % partial derivative of U w.r.t phi
V = zeros(n_phi,n_w);                      % partial derivative of U w.r.t W
U = zeros(n_phi,n_w);                      % U grid values

%% Initial Boundary Conditions
U(:,2)     = sqrt(2*del_w);                           % approximation of U at first del_W increment
V(:,2)     = (1 - del_w)./U(:,2);                     % approximation of V at first del_W increment
Q(:,2)     = 0;                                       % approximation of Q at first del_W increment
Q(1,:)     = 0;                                       % BC's at phi = 0              
Q(n_phi,:) = 0;                                       % BC's at phi = pi
U(1,2)     = (1/3)*(4*U(2,2) - U(3,2));               % BC's at phi = 0
U(n_phi,2) = (1/3)*(4*U(n_phi-1,2) - U(n_phi-2,2));   % BC's at phi = pi
V(1,2)     = (1/3)*(4*V(2,2) - V(3,2));               % BC's at phi = 0
V(n_phi,2) = (1/3)*(4*V(n_phi-1,2) - V(n_phi-2,2));   % BC's at phi = pi

%% Calculation start (Thomas algorithm)
for j = 2:n_w-1
    % initialize coefficients for level j
    A = 1 + (Q(:,j).^2)./(U(:,j).^2 + epsilon);
    B = -(V(:,j).*Q(:,j))./(U(:,j).^2 + epsilon);
    C = (1 + (V(:,j).^2))./(U(:,j).^2 + epsilon);
    D = (2 - Bo.*(U(:,j).*sin(alpha).*cos(phi)' - W(1,j).*cos(alpha))).*((1 + (V(:,j).^2) + ...
        (Q(:,j).^2)./(U(:,j).^2 + epsilon)).^(3/2)) - (1./(U(:,j) + epsilon)).*(1 + (V(:,j).^2) + 2.*(Q(:,j).^2)./(U(:,j).^2 + epsilon));

    % apply Thomas algorithm for V at level j+1
    V(:,j+1) = Thomas_Algo(A, B, C, D, n_phi, del_phi, del_w, Q(:,j), V(:,j), epsilon);

    % update values of Q and U at level j+1
    for k = 2:n_phi-1
        Q(k,j+1) = Q(k,j) + (del_w/(2*del_phi))*(V(k+1,j+1) - V(k-1,j+1));
        U(k,j+1) = U(k,j) + (del_w/2)*(V(k,j) + V(k,j+1));
    end

    % boundary conditions on U at level j+1
    U(1,j+1)     = (1/3)*(4*U(2,j+1) - U(3,j+1));
    U(n_phi,j+1) = (1/3)*(4*U(n_phi-1,j+1) - U(n_phi-2,j+1));

    % checking overflow and stopping iteration
    err = abs(b*(U(n_phi,j+1) - U(n_phi,j)).*cos(phi(1,n_phi))'.*1000);
    if err > 1
        clip_idx = j;
        break;
    end

end

x = b.*U(:,1:clip_idx).*cos(phi)'.*1000;
y = b.*U(:,1:clip_idx).*sin(phi)'.*1000;
z = b.*W(1,1:clip_idx).*1000;

% plot shape
plot_droplet(x,y,z,n_phi,clip_idx, Bo, alpha);

%% Function Definition
function out = Thomas_Algo(A, B, C, D, n_phi,del_phi, del_w, Q, V, epsilon)
    % function to solve system of equation Mx = g, where M is the
    % tridiagonal matrix formed by the combination of coefficients A and B, and
    % g is the vector formed by the combination of coefficients C, D and
    % vector functions Q and V

    % coefficient initialization
    J = zeros(n_phi,n_phi);
    d = zeros(n_phi,1);
    a = zeros(n_phi,1);
    b = zeros(n_phi,1);
    c = zeros(n_phi,1);
    e = zeros(n_phi,1);

    % Value assignment for generating vector g
    for k = 2:n_phi-1
        J(k,k-1) = C(k-1)/(2*del_phi);
        J(k,k+1) = -C(k+1)/(2*del_phi);
        d(k)     = D(k);
        e(k)     = A(k)/del_w;
    end
    % vector g
    g = J*Q - d + e.*V;
    
    % value assignment for generating tridiagonal coefficients
    for k = 1:n_phi
        if (k > 1)
            if(k == n_phi)
                a(k) = -1;
            else
                a(k) = -B(k-1)/del_phi;
            end
        end

        if (k < n_phi)
            if(k == 1)
                c(k) = 1;
            else
                c(k) = B(k+1)/del_phi;
            end
        end

        if (k == 1)
            b(k) = -1;
        elseif (k == n_phi)
            b(k) = 1;
        else
            b(k) = A(k)/del_w;
        end
    end

    % TDMA solver iterations
    % forward elimination
    for k = 2:n_phi
        ratio = a(k)/(b(k-1) + epsilon);
        b(k)  = b(k) - c(k-1)*ratio;
        g(k)  = g(k) - g(k-1)*ratio;
    end

    % backward substitution
    out = b;
    out(n_phi) = g(n_phi)/(b(n_phi) + epsilon);
    for k = (n_phi-1):-1:1
        out(k) = (g(k) - c(k)*out(k+1))/(b(k) + epsilon);
    end
end

function plot_droplet(x, y, z, n_phi, n_w, Bo, alpha)
    zz = zeros(n_phi,n_w);
    for k = 1:n_phi
        zz(k,:) = z;
    end
    fh = figure();
    fh.WindowState = 'maximized';
    surf(x, y, zz, 'EdgeAlpha',0.05);
    colormap([0.5 0.5 0.5]);
    hold on;
    surf(x, -y, zz,'EdgeAlpha',0.05);
    colormap([0.5 0.5 0.5]);
    view(0,0);
    zmax = max(zz,[], 'all');
    zlim([-0.5, zmax+0.2]);
    xlabel('X (mm)', 'FontWeight','bold');
    ylabel('Y (mm)', 'FontWeight','bold');
    zlabel('Z (mm)', 'FontWeight','bold');
    set(gca, 'Zdir', 'reverse', 'Fontsize', 30,'Fontweight', 'bold');
    %title("Droplet surface profile for Bo = " + num2str(Bo) + " and tilt angle = " + num2str(alpha*180/pi) + " deg");
    grid('off');
    %axis('off');
    
end



