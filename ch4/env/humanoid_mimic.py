import pybullet
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from mimic.envs.robot import Humanoid
import numpy as np

from json import load
from gym import spaces

class HumanoidMimicEnv(WalkerBaseBulletEnv):
    def __init__(self, serial='02_01', robot=None, render=False):
        if robot is None:
            self.robot = Humanoid()
        else:
            self.robot = robot
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space

        self._load_data(serial)

    def _load_data(self, serial):
        data = np.load('mimic/data/'+serial+'.npz')
        self.frameTime = data['frameTime']
        self._max_episode_steps = data['frames']
        self.cmu_quat = data['quat']
        self.cmu_vec = data['vec']

    def reset(self):
        if (self.stateId >= 0):
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = \
            self.robot.addToScene(self._p, self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], \
            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()

        self.current_frame = 0
        self.init_state = r
        
        return r


    def step(self, a):
        if not self.scene.multiplayer:
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()
        obsvec, obsquat = self.get_vec_quat()

        reward = 0
        done = self.current_frame >= self._max_episode_steps

        def vec_dist(arr1, arr2):
            return np.sum([np.linalg.norm(a-b) for (a,b) in zip(arr1, arr2)])

        if not done:
            cmuvec = self.cmu_vec[self.current_frame]
            cmuquat = self.cmu_quat[self.current_frame]

            loss_pos = vec_dist(obsvec,cmuvec)
            loss_quat = vec_dist(obsquat,cmuquat)
            reward = -(loss_pos + loss_quat)

            self.current_frame += 1
        
        return state, reward, done, {}

    def get_vec_quat(self):
        pos = self.robot.get_joint_position()
        body_quat = self.robot.robot_body.current_orientation()

        '''
            waist -> right_hip
            right_hip -> right_knee
            waist -> left_hip
            left_hip -> left_knee
            waist -> right_shoulder
            right_shoulder -> right_elbow
            waist -> left_shoulder
            left_shoulder -> left_elbow
            waist -> head
            right_knee -> right_foot
            left_knee -> left_foot
            right_elbow -> right_hand
            left_elbow -> left_hand
        '''
        idx = [(9,0), (0,1), (9,2), (2,3),
                (9,4), (4,5), (9,6), (6,7),
                (9,8), (1, 10), (3, 11), (5,12), (7,13)]
        vec = [pos[t] - pos[f] for f,t in idx]
        normalized = np.array([np.divide(v,np.linalg.norm(v)) for v in vec])

        #print(pos)

        return normalized, body_quat
