# Phase field fracture implementation in FEniCS    
# The code is distributed under a BSD license     
      
# If using this code for research or industrial purposes, please cite:
# Hirshikesh, S. Natarajan, R. K. Annabattula, E. Martinez-Paneda.
# Phase field modelling of crack propagation in functionally graded materials.
# Composites Part B: Engineering 169, pp. 239-248 (2019)
# doi: 10.1016/j.compositesb.2019.04.003
      
# Emilio Martinez-Paneda (mail@empaneda.com)
# University of Cambridge

# Preliminaries and mesh
from dolfin import *
# mesh = Mesh('mesh.xml')

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

# Introduce manually the material parameters
# Gc =  2.7
# l = 0.015
# lmbda = 121.1538e3
# mu = 80.7692e3


Gc, l, lmbda, mu, eta_eps =  1.721/ (100e3), 0.015, 0, 1, 1.e-3

# Constituive functions
def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))
def psi(u):
    return 0.5*(lmbda+mu)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+\
           mu*inner(dev(epsilon(u)),dev(epsilon(u)))		
def H(uold,unew,Hold):
    return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
		
# Boundary conditions
top = CompiledSubDomain("near(x[1], 0.5) && on_boundary")
bot = CompiledSubDomain("near(x[1], -0.5) && on_boundary")

left = CompiledSubDomain("near(x[0], -0.5) && on_boundary")
right = CompiledSubDomain("near(x[0], 0.5) && on_boundary")


def Crack(x):
    return abs(x[1]) < 10e-03 and abs(x[0]) <= 0.25

u_Rx = Expression("t", t=0.0, degree=1)
u_Lx = Expression("-t", t=0.0, degree=1)



bc_bot = DirichletBC(W.sub(1), Constant(0.0), bot)
bc_top = DirichletBC(W.sub(1), Constant(0.0), top)
bc_right = DirichletBC(W.sub(0), u_Rx, right)
bc_left = DirichletBC(W.sub(0), u_Lx, left)

bc_u = [bc_bot , bc_top, bc_right, bc_left]

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


fileUU = File("./ResultsDir/displacement2.pvd")

parameters['allow_extrapolation'] = False
mesh2 = RectangleMesh.create([Point(-0.5, -0.5),Point(0.5, 0.5)],[30,30],CellType.Type.quadrilateral)
V2 = VectorFunctionSpace(mesh2, "CG", 1)


TS = TensorFunctionSpace(mesh, "DG", 0)
stress_plot = Function(TS)
filestress = File("./ResultsDir/stress.pvd")

# Initialization of the iterative procedure and output requests
t = 0
u_r = 1
deltaT  = 1e-3
tol = 1e-3
conc_f = File ("./ResultsDir/phi.pvd")
conc_u = File ("./ResultsDir/disp.pvd")
fname = open('ForcevsDisp.txt', 'w')

# Staggered scheme
while t<= 0.012:
    t += deltaT
    # if t >=0.7:
    #     deltaT = 0.0001
    
    u_Rx.t=t
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

                Traction = dot(sigma(unew),n)
                fy = Traction[1]*ds(1)
                fname.write(str(t*u_r) + "\t")
                fname.write(str(assemble(fy)) + "\n")
                
                u2 = project(unew, V2)
                u2.rename("u2","displacement2")
                fileUU << u2
                
                stress_plot = project(sigma(unew), TS)
                stress_plot.rename("stress", "stress")
                filestress  << stress_plot
	    	    
fname.close()
print ('Simulation completed') 
