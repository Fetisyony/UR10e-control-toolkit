"""Basic tools for robot control.
"""


import urx
# pip install git+https://github.com/jkur/python-urx


class OperateRobot:

    def __init__(self, ip):
        self.rob = urx.Robot(ip)

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def movel(self, point: dict):
        self.rob.movel((point["x"], point["y"], point["z"],
                       point["rx"], point["ry"], point["rz"]), 0.2, 0.2)

    def movej(self, point: dict):
        self.rob.movej((point["w1"], point["w2"], point["w3"],
                       point["w4"], point["w5"], point["w6"]), 0.2, 0.2)

    def getl(self):
        return self.rob.getl()

    def getj(self):
        return self.rob.getj()

    def close(self):
        self.rob.close()

    def open_gripper(self):
        self.rob.send_program('set_tool_digital_out(0, True)')
        self.rob.send_program('set_tool_digital_out(1, False)')

    def close_gripper(self):
        self.rob.send_program('set_tool_digital_out(0, False)')
        self.rob.send_program('set_tool_digital_out(1, True)')
