from pylab import *
from numpy import *
from arnoldcat import *
from numba import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches, path
inner_radius = 3.0

@jit(nopython=True)
def map_state_to_torus(x,y):
    phi = 2*pi*x
    theta_mapped = 2*pi*y
    z_mapped = sin(phi)
    r_mapped = inner_radius + \
            cos(phi)
    x_mapped = r_mapped*cos(theta_mapped)
    y_mapped = r_mapped*sin(theta_mapped)
    return x_mapped, y_mapped, z_mapped


@jit(nopython=True)
def map_tangent_to_torus(u,v):
    x = u[0]
    y = u[1]
    n = u.shape[1]
    vtheta = v[0]
    vphi = v[1]
    phi = 2*pi*x
    theta_mapped = 2*pi*y
    z_mapped = sin(phi)
    r_mapped = inner_radius + \
            cos(phi)
    x_mapped = r_mapped*cos(theta_mapped)
    y_mapped = r_mapped*sin(theta_mapped)
    
    ddx_ddphi_mapped = -x_mapped*z_mapped/r_mapped 
    ddx_ddtheta_mapped = -y_mapped/r_mapped/r_mapped
    ddy_ddphi_mapped = -y_mapped*z_mapped/r_mapped/r_mapped
    ddy_ddtheta_mapped = x_mapped/r_mapped/r_mapped
    ddz_dphi_mapped = cos(phi)

    v_mapped = zeros((3,n))
    v_mapped[0] = ddx_ddtheta_mapped*vtheta + \
            ddx_ddtheta_mapped*vphi 

    v_mapped[1] = ddy_ddtheta_mapped*vtheta + \
            ddy_ddtheta_mapped*vphi 

    v_mapped[2] = ddz_dphi_mapped*vphi 

    return v_mapped

@jit(nopython=True)
def map_line_to_torus(phi0, theta0, m, n=100):
    phi0 /= 2*pi
    theta0 /= 2*pi
    phi = linspace(0, 1., n)
    theta = (m*(phi - phi0) + theta0)%1
    x_line, y_line, z_line = map_state_to_torus(phi,theta)
    return x_line, y_line, z_line    

@jit(nopython=True)
def get_primal_trajectory(solver, u0, s, n):
    state_dim = u0.size
    u = zeros((n, state_dim))
    u[0] = u0
    for i in range(1,n):
        u[i] = solver.primal_step(u[i-1],s,1)
    return u


def draw_state_on_torus(u):
    n = u.shape[0]
    u = u.T
    x, y, z = map_state_to_torus(u[0], u[1])
    fig = figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(labelsize=14)
    ax.scatter(x, y, z, c="k")
    return fig, ax


def draw_tangent_on_torus(u,v,c="red"):
    n = u.shape[0]
    state_dim = u.shape[1]
    u = u.T
    v = v.T
    x, y, z = map_state_to_torus(u[0], u[1])
    fig = figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(labelsize=14)
    ax.scatter(x, y, z, c="k")
    v = map_tangent_to_torus(u,v)
    ax.quiver(x[::10], y[::10], z[::10], v[0,::10], \
                v[1,::10], v[2,::10], color=c, normalize=True)

    phi, theta = mgrid[0.0:2.0*pi:100j, 0.0:2.0*pi:100j]
    r = inner_radius + cos(phi)
    x = r*cos(theta)
    y = r*sin(theta)
    z = sin(phi)
    ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='gray', alpha=0.5, linewidth=0)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])    
    ax.zaxis.set_ticklabels([])
    ax.view_init(60,None)
    return fig, ax


def draw_torus():
    fig = figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    phi, theta = mgrid[0.0:2.0*pi:100j, 0.0:2.0*pi:100j]
    r = inner_radius + cos(phi)
    x = r*cos(theta)
    y = r*sin(theta)
    z = sin(phi)
    ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='gray', alpha=0.5, linewidth=0)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])    
    ax.zaxis.set_ticklabels([])
    ax.view_init(65,None)
    r_label = inner_radius + 2.5
    phi_label = linspace(0.0,pi/6.0,10)
    x_label = r_label*cos(phi_label)
    y_label = r_label*sin(phi_label)
    theta_vertices = zeros((10,2))
    theta_vertices[:,0] = x_label
    theta_vertices[:,1] = y_label
    theta_path = path.Path(theta_vertices)
    fig.patches.insert(0,patches.PathPatch(theta_path))
    return fig, ax
   
def draw_local_invariant_manifolds(phi,theta):
    fig, ax = draw_torus()
    eig_vals, eig_vecs = linalg.eig(array([\
            2,1,1,1]).reshape(2,2))
    ms = eig_vecs[1,1]/eig_vecs[1,0]
    mu = eig_vecs[0,1]/eig_vecs[0,0]
    x_line, y_line, z_line = map_line_to_torus( \
            phi, theta, ms)
    ax.plot(x_line,y_line,z_line,lw=2.5,c="b")
    x_line, y_line, z_line = map_line_to_torus( \
            phi, theta, mu)
    ax.plot(x_line,y_line,z_line,c="r",lw=2.5)


if __name__ == "__main__":

    solver = Solver()
    n = 1000
    u0 = rand(2)
    u = get_primal_trajectory(solver, u0, solver.s0, \
            n)
    l, v = solver.get_lyapunov_exponents_vectors()
    v0 = v[0]
    v0 = tile(v0, (n, 1))
    #fig, ax = draw_tangent_on_torus(u,v0)


