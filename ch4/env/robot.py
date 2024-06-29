# Original is bullet3/examples/pybullet/gym/pybullet_envs/robot_locomotors.py

from pybullet_envs.robot_locomotors import WalkerBase

import numpy as np


class Humanoid(WalkerBase):
  self_collision = True
  foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

  def __init__(self):
    WalkerBase.__init__(self,
                        'humanoid_symmetric.xml',
                        'torso',
                        action_dim=17,
                        obs_dim=44,
                        power=0.41)
    # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

  def robot_specific_reset(self, bullet_client):
    WalkerBase.robot_specific_reset(self, bullet_client)
    self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
    self.motor_power = [100, 100, 100]
    self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
    self.motor_power += [100, 100, 300, 200]
    self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
    self.motor_power += [100, 100, 300, 200]
    self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
    self.motor_power += [75, 75, 75]
    self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
    self.motor_power += [75, 75, 75]
    self.motors = [self.jdict[n] for n in self.motor_names]

    if self.random_yaw:
      position = [0, 0, 0]
      orientation = [0, 0, 0]
      yaw = self.np_random.uniform(low=-3.14, high=3.14)
      if self.random_lean and self.np_random.randint(2) == 0:
        cpose.set_xyz(0, 0, 1.4)
        if self.np_random.randint(2) == 0:
          pitch = np.pi / 2
          position = [0, 0, 0.45]
        else:
          pitch = np.pi * 3 / 2
          position = [0, 0, 0.25]
        roll = 0
        orientation = [roll, pitch, yaw]
      else:
        position = [0, 0, 1.4]
        orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
      self.robot_body.reset_orientation(orientation)
    else:
      position = [0, 0, 1.3]
    self.robot_body.reset_position(position)
    
    self.initial_z = 1.3 # change from original (0.8)

  random_yaw = False
  random_lean = False

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    force_gain = 1
    for i, m, power in zip(range(17), self.motors, self.motor_power):
      m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

  def alive_bonus(self, z, pitch):
    return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

  def get_joint_position(self):
    #joint_name = jointInfo[1] ... self.motor_names
    #part_name = jointInfo[12] ... link0_n
    
    jointInfo = [j._p.getJointInfo(j.bodies[j.bodyIndex], j.jointIndex) for j in self.ordered_joints]
    linkName= ['link0_7','link0_11', # right_hip, right_knee
              'link0_14', 'link0_18', # left_hip, left_knee
              'link0_21', # right_shoulder
              'link0_24', # right_elbow
              'link0_26', # left_shoulder
              'link0_29', # left_elbow
              'torso', 'lwaist', 'right_foot', 'left_foot', # torso for head
              'right_lower_arm', 'left_lower_arm'] # for hand

    parentPos = np.array([j[-3] for j in jointInfo if j[12].decode() in linkName] + \
                        [[0, 0, .19,], [-.01, 0, -0.260], [0,0,0], [0,0,0],
                        [.18, .18, .18], [.18, -.18, .18]]) # from humanoid_symmetric.xml

    pos = np.array([self.parts.get(l).get_position() for l in linkName]) + parentPos
    
    return pos

  def is_landing(self):
    for i, f in enumerate(self.feet):
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      if (self.ground_ids & contact_ids):
        self.feet_contact[i] = 1.0
      else:
        self.feet_contact[i] = 0.0
    return 1.0 in self.feet_contact