import mujoco
import numpy as np
from typing import Literal


class ElasticBand:

    def __init__(self, property: Literal['spring', 'elastic'] = 'elastic', enable=True):
        self.stiffness = 300
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = enable
        self.property = property

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        if self.property =="spring":
            f = (self.stiffness *
                (distance - self.length) - self.damping * v) * direction
        else:
            # 只有当距离大于自然长度时才产生拉力
            if distance > self.length:
                f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
            else:
                # 距离小于等于自然长度时不产生力
                f = np.zeros_like(direction)
        # f[0]=0
        # f[1]=0
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_UP:
            self.length -= 0.1
        if key == glfw.KEY_DOWN:
            self.length += 0.1
        if key == glfw.KEY_ESCAPE:
            self.enable = not self.enable
