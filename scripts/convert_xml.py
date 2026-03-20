import mujoco

# model = mujoco.MjModel.from_xml_path("resources/dex_v3/urdf/dex_v3_mujoco.urdf")

# mujoco.mj_saveLastXML("resources/dex_v3/urdf/dex_v3_mujoco_v2.xml",model)

model = mujoco.MjModel.from_xml_path("resources/evt2/urdf/evt2_mujoco.urdf")

mujoco.mj_saveLastXML("resources/evt2/urdf/evt2.xml",model)
