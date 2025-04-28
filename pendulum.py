import pybullet as p
import time
import pybullet_data
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

guiFlag = False

dt = 1/240 # pybullet simulation step
q0 = 0.5   # starting position (radian)
g = 10     # m/s^2
L = 0.8    # m
m = 1      # kg
physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-g)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./simple.urdf.xml", useFixedBase=True)

# get rid of all the default damping forces
# think of it as imagined "air drag"
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=q0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 5 # seconds
logTime = np.arange(0, 5, dt)
sz = len(logTime)
logThetaSim = np.zeros(sz)
idx = 0
for t in logTime:
    logThetaSim[idx] = p.getJointState(boxId, 1)[0]
    p.stepSimulation()
    idx += 1
    if guiFlag:
        time.sleep(dt)
p.disconnect()

def right_part(x, t):
    return np.array([x[1],
                     -g/L * np.sin(x[0])])

# substitute with pybullet-based integration method
theta = odeint(func=right_part,
               y0=[q0, 0],
               t=logTime)

# l2-norm sqrt(avg sum of squares of diffs)
# linf max(abs(diff))

logThetaInt = theta[:,0]

plt.plot(logTime, logThetaSim, 'b', label="Sim Pos")
plt.plot(logTime, logThetaInt, 'r', label="Int Pos")
plt.grid(True)
plt.legend()
plt.show()

# dt = 0.1
# dx = f(x,t) t = [0, 0.1, 0.2]
# x(t)
# dx/dt = f(x,t)
# dx = dt * f(x,t)
# x[n+1] - x[n] = dt * f(x,t)
# x[n+1] = x[n] + dt*f(x,t) # Euler method

# lim(dx/dt) dt -> 0