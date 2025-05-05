import pybullet as p
import time
import pybullet_data
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

guiFlag = False

dt = 1/240 # pybullet simulation step
th0 = 0.5  # starting position (radian)
thd = 1.0  # desired position (radian)
kp = 40.0  # proportional coefficient
ki = 40.0
kd = 20.0
g = 10     # m/s^2
L = 0.8    # m
m = 1      # kg
f0 = 10    # applied const force
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
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 5 # seconds
logTime = np.arange(0, 5, dt)
sz = len(logTime)
logThetaSim = np.zeros(sz)
logVelSim = np.zeros(sz)
idx = 0
e_int = 0
e_prev = 0
for t in logTime:
    th = p.getJointState(boxId, 1)[0]
    vel = p.getJointState(boxId, 1)[1]
    logThetaSim[idx] = th
    e = th - thd
    e_int += e*dt
    de = (e - e_prev) / dt
    e_prev = e
    # PID regulator
    # dth = -kp * e -ki * e_int - kd * de

    # Feedback linearization
    dth = (m*L*L)*(g/L*np.sin(th) - kp*e - kd * vel)

    # p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=dth, controlMode=p.VELOCITY_CONTROL)
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=dth, controlMode=p.TORQUE_CONTROL)
    p.stepSimulation()
    # p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=f0, controlMode=p.TORQUE_CONTROL)
    vel = p.getJointState(boxId, 1)[1]
    logVelSim[idx] = vel

    idx += 1
    if guiFlag:
        time.sleep(dt)
p.disconnect()

def right_part(x, t):
    return np.array([x[1],
                     -g/L * np.sin(x[0]) + f0/(m*L*L)])

# substitute with pybullet-based integration method
theta = odeint(func=right_part,
               y0=[th0, 0],
               t=logTime)

# l2-norm sqrt(avg sum of squares of diffs)
# linf max(abs(diff))

logThetaInt = theta[:,0]

plt.subplot(2,1,1)
plt.plot(logTime, logThetaSim, 'b', label="Sim Pos")
plt.plot([logTime[0], logTime[-1]], [thd, thd], 'r--', label="Ref Pos")
# plt.plot(logTime, logThetaInt, 'r', label="Int Pos")
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(logTime, logVelSim, 'b', label="Sim Vel")
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

# ddth = -g/L * sin(th)
# mL^2*ddth + mgLsin(th) = tau
# ddth = - g/Lsin(th) + tau/(mL^2)
# tau = (mL^2)g/Lsin(th) + u(t) -> ddth = -g/Lsin(th) + ((mL^2)g/Lsin(th) + u(t))/(mL^2)
# ddth = -g/Lsin(th) + g/Lsin(th) + u(t)
# ddth = u(t)
# ddth = kp(th-thd)
# Feedback linearization
# Линеаризация обратной связью