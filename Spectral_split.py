from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

mesh = RectangleMesh(Point(-0.5,-0.5),Point(0.5,0.5), 40, 40)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.5:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.5:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.4:
        cell_markers[cell] = True
# mesh = refine(mesh, cell_markers)
# Define Space
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 0)
p, q = TrialFunction(V), TestFunction(V)
u, v = TrialFunction(W), TestFunction(W)


Gc, l, lmbda, mu, eta_eps =  1.721/ (100e3), 0.015, 0, 1, 1.e-3

# Constituive functions
def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))

def sigma_plotting(u, p):
    return ( (1-p)**2)* (2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u)))    

def trace_eps_positive_square(eps):
    
    mean_eps = 0.5 * (eps[0,0] + eps[1,1])
    det_eps = eps[0,0] * eps[1,1] - eps[0,1] * eps[1,0]

    eigen_val_1 = mean_eps + sqrt(mean_eps**2 - det_eps)
    eigen_val_2 = mean_eps - sqrt(mean_eps**2 - det_eps)
    #print('eigen_val_1 : ', eigen_val_1)
    eig_1_pos = 0.5 * (eigen_val_1 + abs(eigen_val_1))
    eig_2_pos = 0.5 * (eigen_val_2 +abs(eigen_val_2))
    return eig_1_pos**2 + eig_2_pos**2


def psi(u):
    eps = epsilon(u)
    tr_eps = tr(eps)
    first_term = 0.5 * lmbda * ( 0.5* (tr_eps + abs(tr_eps)))**2
    tr_eps_pos_square = trace_eps_positive_square(eps)
    second_term = mu * tr_eps_pos_square
    return first_term + second_term	


def H(uold,unew,Hold):
    return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
		
# Boundary conditions
top = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bot = CompiledSubDomain("near(x[1], -0.5) && on_boundary")

left = CompiledSubDomain("near(x[0], -0.5) && on_boundary")
right = CompiledSubDomain("near(x[0], 0.5) && on_boundary")


def Crack(x):
    return abs(x[1]) < 10e-03 and x[0] <= 0

u_Tx = Expression("t", t=0.0, degree=1)
u_Lx = Expression("-t", t=0.0, degree=1)


bc_bot = DirichletBC(W.sub(1), Constant(0.0), bot)
bc_bot1 = DirichletBC(W.sub(0), Constant(0.0), bot)
bc_top = DirichletBC(W.sub(0), u_Tx, top)


bc_u = [bc_bot , bc_top, bc_bot1]

bc_phi = [DirichletBC(V, Constant(1.0), Crack)]
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Variational form
unew, uold = Function(W), Function(W)
pnew, pold, Hold = Function(V), Function(V), Function(V)
E_du = ((1.0-pold)**2)*inner(grad(v),sigma(u))*dx
E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H(uold,unew,Hold))\
            *inner(p,q)-2.0*H(uold,unew,Hold)*q)*dx
p_disp = LinearVariationalProblem(lhs(E_du), rhs(E_du), unew, bc_u)
p_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), pnew, bc_phi)
solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)

TS = TensorFunctionSpace(mesh, "DG", 0)

stress_plotting = Function(TS)
filestress_plotting = File("./Results/stress_plotting.pvd")

# Initialization of the iterative procedure and output requests
t = 0
u_r = 1
deltaT  = 1e-3
tol = 1e-3
conc_f = File ("./Results/phi.pvd")
conc_u = File ("./Results/disp.pvd")

# Staggered scheme
while t<= 0.012:
    t += deltaT
    # if t >=0.7:
    #     deltaT = 0.0001
    
    u_Tx.t=t
    u_Lx.t=t
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        solver_disp.solve()
        solver_phi.solve()
        err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
        err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
        err = max(err_u,err_phi)
        
        uold.assign(unew)
        pold.assign(pnew)
        Hold.assign(project(psi(unew), WW))

        if err < tol:
		
            print ('Iterations:', iter, ', Total time', t)

            if round(t*1e4) % 10 == 0:
                conc_f << pnew
                conc_u << unew
                
                stress_plotting = project(sigma_plotting(unew, pnew), TS)
                stress_plotting.rename("stress_plotting", "stress")
                filestress_plotting  << stress_plotting

                p = plot(pnew, colorbar = 'False')
                plt.colorbar(p)
                plt.savefig("pnew.eps", format = 'eps')


print ('Simulation completed') 
