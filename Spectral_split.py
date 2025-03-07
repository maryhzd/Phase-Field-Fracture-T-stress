# --------------------------------
#https://home.iitm.ac.in/ratna/codes/phasefield/phasefield/Tension%20Test/Tension.py
# purpose: phase field for fracture
# Tension test 2D
#  Hirshikesh, Sundarajan Natarajan, Ratna Kumar Annabatutla 
#  IITM, Aug 2017
#-------------------------------------
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

from dolfin import Point
#mesh = Mesh('mesh.xml')

# set_log_active(False)
# #==================================
# # mesh (concident nodes) for the crack
# #===================================
# nx = 1
# ny = 1
# num = 1
# mesh = Mesh()
# editor = MeshEditor()
# editor.open(mesh, "triangle", 2, 2)
# editor.init_vertices(10)
# editor.init_cells(8)

# editor.add_vertex(0, Point(0, 0.0))
# editor.add_vertex(1, Point(0.5, 0.0))
# editor.add_vertex(2, Point(0.5, 0.5))
# editor.add_vertex(3, Point(0.0, 0.5))
# editor.add_vertex(4, Point(-0.5, 0.5))
# editor.add_vertex(5, Point(-0.5, 0.0))
# editor.add_vertex(6, Point(-0.5, -0.5))
# editor.add_vertex(7, Point(0.0, -0.5))
# editor.add_vertex(8, Point(0.5, -0.5))
# editor.add_vertex(9, Point(-0.5, 0.0))

# editor.add_cell(0, [0, 1, 3])
# editor.add_cell(1, [1, 2, 3])
# editor.add_cell(2, [0, 3, 4])
# editor.add_cell(3, [0, 4, 5])
# editor.add_cell(4, [0, 9, 7])
# editor.add_cell(5, [6, 7, 9])
# editor.add_cell(6, [0, 7, 8])
# editor.add_cell(7, [0, 8, 1])


# editor.close()
# #--------------------------------------
# # change num_refine for mesh refinement
# #--------------------------------------
# num_refine = 5
# h=0.5**num_refine
# for i in range(num_refine):
#     mesh=refine(mesh)
# print ("number of unknown",mesh.num_vertices() )
# print ("number of elements",mesh.num_cells() )
# plot(mesh)
#interactive()

# ---------------------------
# get top boundary nodes for Reaction force 
# calculations
#----------------------------

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


def num_nem(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result
class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
                tol = 1E-14
                return on_boundary and near(x[1], 0.5, tol)

topBoundary = TopBoundary()
ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

topBoundary.mark(ff,1)

It_mesh = SubsetIterator(ff, 1)
a = []
b = []
for c in It_mesh:
     for v in vertices(c):
         a.append(v.midpoint().x())
         b.append(v.midpoint().y())
         
dd = num_nem(a)
da = num_nem(b)
#------------------
# check node number 
#------------------
print (dd)
print (da)


#------------------------------------------------
#          Define Space
#------------------------------------------------

V = FunctionSpace(mesh,'CG',1)
W = VectorFunctionSpace(mesh,'CG',1)

p , q = TrialFunction(V), TestFunction(V)
u , v = TrialFunction(W), TestFunction(W)
#------------------------------------------------
#           Parameters
#------------------------------------------------
l_fac = 2
hm = mesh.hmin()
l_o = l_fac*hm
print (l_o)




Gc, l, lmbda, mu, eta_eps =  1.721/ (100e3), 0.015, 0, 1, 1.e-3


#------------------------------------------------
#           Classes
#------------------------------------------------

def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return 2.0*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(2)
def en_dens(u):
    str_ele = 0.5*(grad(u) + grad(u).T)
    IC = tr(str_ele)
    ICC = tr(str_ele * str_ele)
    return (0.5*lmbda*IC**2) + mu*ICC

def eps(u):
    return sym(grad(u))

kn = lmbda + mu
def hist(u):
    return 0.5*kn*( 0.5*(tr(eps(u)) + abs(tr(eps(u)))) )**2 + mu*tr(dev(eps(u))*dev(eps(u)))



#---------------------------
# Boundary conditions
#---------------------------


class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]-0.5) < tol and on_boundary

class bottom(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]+0.5) < tol and on_boundary
class Middle(SubDomain): 
    def inside(self,x,on_boundary):
        tol = 1e-3
        return abs(x[1]) < 10e-3 and abs(x[0]) <= 0.25
    
class right(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[0]-0.5) < tol and on_boundary  
    
    
class left(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[0]+0.5) < tol and on_boundary     

middle = Middle()    
Top = top()
Bottom = bottom()
Right = right()
Left = left()


u_Rx = Expression("t", t=0.0, degree=1)
u_Lx = Expression("-t", t=0.0, degree=1)


bc_bot = DirichletBC(W.sub(1), Constant(0.0), Bottom)
bc_top = DirichletBC(W.sub(1), Constant(0.0), Top)
bc_right = DirichletBC(W.sub(0), u_Rx, Right)
bc_left = DirichletBC(W.sub(0), u_Lx, Left)

bc_u = [bc_bot , bc_top, bc_right, bc_left]

bc_phi = [DirichletBC(V,Constant(1.0),middle)]

#----------------------------
# Define variational form
#----------------------------
uold, unew, uconv = Function(W), Function(W), Function(W)
phiold, phiconv = Function(V), Function(V)
E_du =  ( pow((1-phiold),2) + 1e-6)*inner(grad(v), sigma(u))*dx


E_phi = ( Gc*l_o*inner(grad(p),grad(q))+\
            ((Gc/l_o) + 2.0*hist(unew))*inner(p,q)-\
            2.0*hist(unew)*q)*dx
    
u = Function(W)
p = Function(V)
p_disp = LinearVariationalProblem(lhs(E_du),rhs(E_du),u,bc_u)
p_phi = LinearVariationalProblem(lhs(E_phi),rhs(E_phi),p, bc_phi)

solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)
t = 0
max_load = 0.007
deltaT  = 1e-3
ut = 1
# -----------------------------------------
# Data from Ref: Ambati et. al.
#------------------------------------------


Crack_file = File ("./Result/crack.pvd")
Displacement_file = File ("./Result/displacement.pvd")


fileUU = File("./ResultsDir/displacement2.pvd")

parameters['allow_extrapolation'] = False
mesh2 = RectangleMesh.create([Point(-0.5, -0.5),Point(0.5, 0.5)],[30,30],CellType.Type.quadrilateral)
V2 = VectorFunctionSpace(mesh2, "CG", 1)


TS = TensorFunctionSpace(mesh, "DG", 0)
stress_plot = Function(TS)
filestress = File("./Results/stress.pvd")


f = open('Tension.txt', 'a')
while t<= 0.012:
    # if t >=5e-3:
    #     deltaT = 1e-5
    u_Rx.t=t*ut
    u_Lx.t=t*ut
    uold.assign(uconv)
    phiold.assign(phiconv)
    iter = 1
    toll = 1e-3
    err = 1

    while err > toll:
        solver_disp.solve()
        unew.assign(u)
        solver_phi.solve()

        err_u = errornorm(u,uold, norm_type = 'l2', mesh = None)
        err_phi = errornorm(p,phiold, norm_type = 'l2', mesh = None)
        err = max(err_u, err_phi)
        print ('iter', iter,'error', err)
        uold.assign(u)
        phiold.assign(p)
        iter = iter+1
        if err < toll:
            uconv.assign(u)
            phiconv.assign(p)
            
            Crack_file << p
            
            Displacement_file << u
            
            u2 = project(u, V2)
            u2.rename("u2","displacement2")
            fileUU << u2
            
            stress_plot = project(sigma(u), TS)
            stress_plot.rename("stress", "stress")
            filestress  << stress_plot
                
                
                
            print ('solution converges after:', iter)
 
            ux,uy = split(u)
            plot(ux,key = 'ux',title = 'u_dispx')
            plot(uy,key ='uy',title = 'u_dispy')
            plot(p,range_min = 0.,range_max = 1.,key = 'phi',title = 'phi%.4f'%(t))
            
            # calculate reaction force
            # k_matx=assemble(E_du)
            # u_vec=u.vector()
            # F_vec=k_matx*u_vec
            # Ft_func=Function(W,F_vec)
            # f_allY=[]
            # for k in dd:
            #     fy=Ft_func(k,0.5)
            #     #print fy
            #     fyt=fy[1]
            #     #print fyt
            #     f_allY.append(fyt)
            #     #print f_allY
            #     f_allY_s=f_allY
            #     #print f_allY
            #     #f_y_u.append(f_allY_s)
            # b=sum(f_allY_s)
            # print (b)
            # plt.scatter(t,(b*1e-3))
            # plt.pause(0.05)
            # a = [(t,b*1e-3)]
            # np.savetxt(f,a)
            

            
	    	    
    t+=deltaT
            

print ('Simulation Done with no error :)')