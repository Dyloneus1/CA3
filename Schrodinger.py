#%%

'''
In this computing assignment, I will animate the non-linear schrodinger equation
(1D schrodinger equation with the potential equal to the wave function itself)
using various additional linear potential functions. We will start with some heaviside
functions multiplied by some constant. We start with .1 to see mostly non-linear behavior
then increase the value to see more of the effect of the potential. These values are given by the
different values Hval1, Hval2, etc...
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
Hval1 = .1
Hval2 = 50
Hval3 = 1000
N = 100
x = np.zeros(N + 1)
dx = 2/N
dt = 1e-3
#Chebyshev-Guass-Lobato points
for j in range(N + 1):
    x[j] = np.cos(j*np.pi/N)
t = np.linspace(0,1,num=int(1/dt))
print(len(t))
#Define the heaviside function for a given Hval
def heaviside(x,Hval):
    if x > 0:
        return Hval
    else:
        return 0
    
#Create the heaviside functions
H1 = np.zeros(N + 1)
H2 = np.zeros(N + 1)
H3 = np.zeros(N + 1)
for i in range(N + 1):
    H1[i] = heaviside(x[i],Hval1)
for i in range(N + 1):
    H2[i] = heaviside(x[i],Hval2)
for i in range(N + 1):
    H3[i] = heaviside(x[i],Hval3)

#Plot the potential
infbound = 1000
plt.plot(x,H1,c='green')
plt.plot([-1,-.999],[infbound,0],c='green')
plt.plot([.999,1],[infbound,Hval1],c='green')

#%%
def cbar(j,n):
    if (j == n or j == 0):
        return 2
    else:
        return 1
'''
Next we want to define the derivative matrix in accordance with Eq. 13.40 in the notes.
We start with the most exclusive condition, if j does not equal k. If this is false, logically
j is equal to k, so we just put the condition that k < N since this also means j < N. The other conditions
follow from this logic.
'''

c2 = []

def d(j,k,x,n):
    if j != k:
        return cbar(j,n)*((-1)**(j+k))/(cbar(k,n)*(x[j] - x[k]))
    elif (k > 0 and k < n):
        return -x[k]/(2*(1-x[k]**2))
    elif k == n:
        return -(1 + 2*(n)**2)/6
    elif k == 0:
        return (1 + 2*(n)**2)/6
    
'''
Forming the derivative matrix
'''
derivmatrix = np.zeros((N+1,N+1))

for j in range(N + 1):
    for k in range(N + 1):
        derivmatrix[j][k] = d(j,k,x,N)

'''
Form a secondary second derivative matrix that ensures the boundary conditions are met in the R-K scheme
'''
secondderivmatrix = np.matmul(derivmatrix,derivmatrix)
secondderivmatrixrungekutta = secondderivmatrix.copy()
secondderivmatrixrungekutta[0] = np.zeros(N+1)
secondderivmatrixrungekutta[-1] = np.zeros(N+1)

# %%
'''
R-K to get the first two terms. Note that "H" here can also be any function of x.
'''

noH = np.zeros(N+1)

def rungekutta(A,H,dt):
    k1 = 1j*dt*(np.matmul(secondderivmatrixrungekutta,A) + 2*np.matmul(derivmatrix,A) + (A*np.conjugate(A) + H)*(A))
    k2 = 1j*dt*(np.matmul(secondderivmatrixrungekutta,(A + k1)) + 2*np.matmul(derivmatrix,A + k1) + (((A + k1)*np.conjugate(A + k1)) + H)*(A + k1))
    return A + (k1 + k2)/2

A0 = np.exp(-100*(x**2))
A1 = rungekutta(A0,noH,dt)
Aarray = [A0,A1]

plt.plot(x,A0)
plt.plot(x,np.real(A1))
plt.plot(x,np.imag(A1))

# %%

'''
Invert the LHS of the matrix equation that meets the B.c.s.
'''
matrixToInvert = np.identity(N+1) - (1j*dt/2)*secondderivmatrix
matrixToInvert[0][0] = 1
matrixToInvert[-1][-1] = 1
for i in range(N):
    matrixToInvert[0][i + 1] = 0
    matrixToInvert[-1][i] = 0

invertedMatrix = np.linalg.inv(matrixToInvert)
# %%

'''
Get the next timestep of the wavefunction and fill an array with those values.
'''

def getA(k,array,H):
    qlin = np.matmul(np.identity(N + 1) + secondderivmatrix*1j*dt/2,array[k])
    qnonlin = (1j*dt/2)*(3*(array[k]*np.conj(array[k] + H)*array[k]) - (array[k - 1]*np.conj(array[k - 1] + H)*array[k - 1]))
    q = qlin + qnonlin
    q[0] = 0
    q[-1] = 0
    return np.matmul(invertedMatrix,q)
#%%
for j in range(1,len(t)):
    Aarray.append(getA(j,array=Aarray,H=noH))

#%%

# '''
# Animation
# '''
# import matplotlib.animation as animation
# infbound = 1000
# fig = plt.figure()



# fig, ax = plt.subplots()
# ax.plot(x,H2,c='green')
# ax.plot([-1,-.999],[infbound,0],c='green')
# ax.plot([.999,1],[infbound,Hval2],c='green')
# ax.set_xlabel('x')
# ax.set_ylabel('A')
# plotLine, = ax.plot(x, np.zeros(len(x))*np.NaN, 'r-',label='Real')
# plotLine2, = ax.plot(x, np.zeros(len(x))*np.NaN, 'b-',label='Imaginary')
# plotTitle = ax.set_title("t=0")
# ax.plot(x,np.zeros(len(x)),'.',color='purple',alpha=.1)
# ax.legend()
# ax.set_ylim(-2,2)
# ax.set_xlim(-1.1,1.1)


# def animate(t):
#     pp = np.real(Aarray[int(t/dt)])
#     pp2 = np.imag(Aarray[int(t/dt)])
#     plotLine.set_ydata(pp)
#     plotLine2.set_ydata(pp2)
#     plotTitle.set_text(f"t = {t:.3f}")
#     return [plotLine,plotTitle]



# ani = animation.FuncAnimation(fig, func=animate, frames=np.arange(0, 1, dt), blit=True)
# plt.show()
# writergif = animation.PillowWriter(fps=30)
# ani.save('schrodinger_50_h_less_n.gif',writer=writergif)
# %%

'''
Next, let's try something more interesting. What if we could have H vary with time?
This would be easy to implement. All we would need to do is define such an H and change
"H" to "H[k]" or "H[k-1]" in the "getA" function. Lets try a heaviside function that decays
with time from 100 to 0. We create a matrix H(x,t). The value of H for x > 0 will be
100, then 100 - 100*1/N+1, 100 - 100*2/N+1, ... , 100 - 100*(N+1)/N+1 = 0.
'''

Hmat = np.zeros((len(t),N+1))
Hinit = 90
for i in range(len(t)):
    for j in range(N+1):
        Hmat[i][j] = heaviside(x[j],Hval=Hinit*np.cos(i*np.pi/100))

for i in range(len(t)):
    if i % 3 == 0:
        plt.plot(x,Hmat[i], label='t = %.2f'%(i*dt))
plt.xlabel("x")
plt.ylabel("H(x)")
plt.title("Time varying H function")
plt.legend()

# %%

'''
We must redefine our R-K scheme
'''

def rungekutta(A,H,dt):
    k1 = 1j*dt*(np.matmul(secondderivmatrixrungekutta,A) + (A*np.conjugate(A) + H)*(A))
    k2 = 1j*dt*(np.matmul(secondderivmatrixrungekutta,(A + k1)) + (((A + k1)*np.conjugate(A + k1)) + H)*(A + k1))
    return A + (k1 + k2)/2

A0 = np.exp(-100*(x + .7)**2)
A1 = rungekutta(A0,Hmat[0],dt)

plt.plot(x,np.real(A0))
plt.plot(x,np.real(A1))

'''
Now we redefine getA
'''

def getA(k,array,H):
    qlin = np.matmul(np.identity(N + 1) + secondderivmatrix*1j*dt/2,array[k])
    qnonlin = (1j*dt/2)*(3*(array[k]*np.conj(array[k] + H[k])*array[k]) - (array[k - 1]*np.conj(array[k - 1] + H[k - 1])*array[k - 1]))
    q = qlin + qnonlin
    q[0] = 0
    q[-1] = 0
    return np.matmul(invertedMatrix,q)
#%%

Tvaryarray = [A0,A1]
for j in range(1,len(t)):
    Tvaryarray.append(getA(j,array=Tvaryarray,H=Hmat))

#%%

# '''
# Animation
# '''
# import matplotlib.animation as animation
# fig = plt.figure()



# fig, ax = plt.subplots()
# ax.set_xlabel('x')
# ax.set_ylabel('A')
# plotLine, = ax.plot(x, np.zeros(len(x))*np.NaN, 'r-',label="Real")
# plotLine2, = ax.plot(x, np.zeros(len(x))*np.NaN, 'b-',label='Imaginary')
# plotLine3, = ax.plot(x, Hmat[0], 'g-',label='V(x,t)')
# plotTitle = ax.set_title("t=0, V_R = 90")
# ax.legend()
# ax.set_ylim(-4,4)
# ax.set_xlim(-1.1,1.1)


# def animate(t):
#     pp = np.real(Tvaryarray[int(t/dt)])
#     pp2 = np.imag(Tvaryarray[int(t/dt)])
#     pp3 = Hmat[int(t/dt)]
#     plotLine.set_ydata(pp)
#     plotLine2.set_ydata(pp2)
#     plotLine3.set_ydata(pp3)
#     plotTitle.set_text("t = {:.2f}, V_R = {:.2f}".format(t,Hinit*np.cos((t/dt)*np.pi/100)))
#     #ax.relim() # use if autoscale desired
#     #ax.autoscale()
#     return [plotLine,plotTitle]



# ani = animation.FuncAnimation(fig, func=animate, frames=np.arange(0, 1, dt), blit=True)
# plt.show()
# writergif = animation.PillowWriter(fps=30)
# ani.save('schrodinger_driving_100_V.gif',writer=writergif)
# %%
