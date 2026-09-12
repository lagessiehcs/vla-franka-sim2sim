import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("../../assets/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)