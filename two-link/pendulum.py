import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

guiFlag = True

dt = 1 / 240
th0 = 0.5
thd = 1.0
kp = 40.0
ki = 40.0
kd = 20.0
g = 10
L = 0.8
L1 = L
L2 = L
m = 1
f0 = 10

xd = 0.5
zd = 1

physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
boxId = p.loadURDF("./two-link.urdf.xml", useFixedBase=True)

p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

pos0 = p.getLinkState(boxId, 4)[0]
X0 = np.array([[pos0[0]], [pos0[2]]])

maxTime = 5
logTime = np.arange(0, maxTime, dt)
sz = len(logTime)
logXsim = np.zeros(sz)
logZsim = np.zeros(sz)
idx = 0
T = 2
for t in logTime:
    th1 = p.getJointState(boxId, 1)[0]
    vel = p.getJointState(boxId, 1)[1]
    th2 = p.getJointState(boxId, 3)[0]
    ve2 = p.getJointState(boxId, 3)[1]

    pos = p.getLinkState(boxId, 4)[0]
    logXsim[idx] = pos[0]
    logZsim[idx] = pos[2]

    zeroVec = [0.0, 0.0]
    jac_t, _ = p.calculateJacobian(
        bodyUniqueId=boxId,
        linkIndex=4,
        localPosition=[0, 0, 0],
        objPositions=[th1, th2],
        objVelocities=[0, 0],
        objAccelerations=[0, 0]
    )
    jac = np.array([
        [jac_t[0][0], jac_t[0][1]],
        [jac_t[2][0], jac_t[2][1]]
    ])
    jac_inv = np.linalg.pinv(jac)

    X = np.array([[pos[0]], [pos[2]]])
    Xd = np.array([[xd], [zd]])

    s = 1
    if t < T:
        s = (3 / T ** 2) * t ** 2 - 2 / (T ** 3) * t ** 3
    Xd_curr = X0 + s * (Xd - X0)

    vel_d = -100.0 * jac_inv @ (X - Xd_curr)
    vel_d = vel_d.flatten()

    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=[1, 3], targetVelocities=vel_d,
                                controlMode=p.VELOCITY_CONTROL)
    p.stepSimulation()

    idx += 1
    if guiFlag:
        time.sleep(dt)
p.disconnect()

plt.subplot(2, 1, 1)
plt.plot(logTime, logXsim)
plt.subplot(2, 1, 2)
plt.plot(logTime, logZsim)
plt.show()

# Forward Kinematics
# x = -L1*sin(th1) - L2*sin(th1+th2)
# z = H - L1*cos(th1) - L2*cos(th1+th2)

# dx = -L1*cos(th1)*dth1 - L2*cos(th1+th2)*(dth1+dth2)
# dz = L1*sin(th1)*dth1 + L2*sin(th1+th2)*(dth1+dth2)

# dx = (-L1*cos(th1) - L2*cos(th1+th2))*dth1 - L2*cos(th1+th2) * dth2
# dz = (L1*sin(th1) + L2*sin(th1+th2))*dth1 + L2*sin(th1+th2) * dth2
# X = (x,z)'
# Th = (th1, th2)'
# dX = J(Th) * dTh
# dTh = inv(J) * dX
# dX = k(Xd - X)

# parametrization
# X(0) = X0
# X(1) = Xd
# X(s) = X0 + s*(Xd-X0)
# s [0, 1]
# s(t)

# X(0) = X0
# X(T) = Xd
# dX(0) = 0
# dx(T) = 0
