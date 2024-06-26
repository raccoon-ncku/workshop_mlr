import pybullet as p
from time import sleep
import pybullet_data


physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf", basePosition=[0, 0, -2])


robot = p.loadURDF(
    "submodules/rccn_robot_cell/robot_description/rccn_kuka_robot_cell/urdf/rccn_kuka_robot_cell.urdf",
    useFixedBase=0,
    flags=p.URDF_USE_SELF_COLLISION)

useRealTimeSimulation = 0

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)

while 1:
  if (useRealTimeSimulation):
    p.setGravity(0, 0, -10)
    sleep(0.01)  # Time in seconds.
  else:
    p.stepSimulation()