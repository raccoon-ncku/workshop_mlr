import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf", basePosition=[0, 0, -2])
robot = p.loadURDF(
    "submodules/rccn_robot_cell/robot_description/rccn_kuka_robot_cell/urdf/rccn_west_robot.urdf",
    useFixedBase=1)

p.setGravity(0,0,-9.8)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

while True:
    p.stepSimulation()
    # time.sleep(1./240)