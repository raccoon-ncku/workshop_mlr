import pybullet as p
import pybullet_data
from time import sleep

p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf", basePosition=[0, 0, -2])
robot = p.loadURDF(
    "submodules/rccn_robot_cell/robot_description/rccn_kuka_robot_cell/urdf/rccn_west_robot.urdf",
    useFixedBase=1)



cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)

useRealTimeSimulation = 1

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

while 1:
  if (useRealTimeSimulation):
    p.setGravity(0, 0, -10)
    sleep(0.01)  # Time in seconds.
  else:
    p.stepSimulation()