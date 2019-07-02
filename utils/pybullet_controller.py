import numpy as np
import pybullet


class CController(object):
    """Abstract controller class

    Attributes:
        err (np.array): Current error values for each of the state dimensions of the controller.
        control_type (str): Possible values: ['joint_pos', 'joint_vel', 'joint_eff', 'cart_pos', 'cart_vel', 'cart_eff']
    """

    def __init__(self, control_type='cart_vel'):
        """
        :param control_type: Possible values: ['joint_pos', 'joint_vel', 'joint_eff', 'cart_pos', 'cart_vel', 'cart_eff']
        """
        self.err = 0
        self.control_type = control_type

    def get_command(self, state, reference):
        """Obtain the control action for the provided current and desired states

        :param state: current state
        :param reference: desired state
        :return: control action
        """
        pass

    def reset(self):
        """
        Resets the controller to its initial state

        """
        self.err = 0


class CPIDController(CController):
    """Generic PID controller class

    Attributes:
        Kp (float): Proportional term gain
        Kd (float): Derivative term gain
        Ki (float): Integral term gain
        iClamp (float): Max value than the integral error might acquire
        err_int (np.array): Accumulated integral error values for each of the state dimensions of the controller.
    """

    def __init__(self, Kp=1, Ki=0, Kd=0, iClamp=0):
        """
        :param Kp (float): Proportional term gain
        :param Kd (float): Derivative term gain
        :param Ki (float): Integral term gain
        :param iClamp (float): Max value than the integral error might acquire
        """
        super(CPIDController, self).__init__()
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.iClamp = iClamp
        self.err_int = []

    def get_command(self, state, ref):
        """Obtain the control action for the provided current and desired states

        Control action computed by:
            $u(t) = K_p e(t) + K_d \frac{de(t)}{dt} + K_i \int_{0}^{t} e(t) dt$

        Where e(t) = ref - state

        :param state: Current state of the controlled variables
        :param ref: Desired state of the controlled variables
        :return: Control action computed as: $u(t) = K_p e(t) + K_d \frac{de(t)}{dt} + K_i \int_{0}^{t} e(t) dt$
        """

        # Obtain current error value
        err = ref - state

        # Compute error derivative using the last stored error value
        err_dot = err - self.err

        # Accumulate error into the integral error variable
        if len(self.err_int) == 0:
            self.err_int = err
        else:
            self.err_int = self.err_int + err
            self.err_int = np.clip(self.err_int, -self.iClamp, self.iClamp)

        # Updated controller error
        self.err = err

        # Obtain the control action using the PID control equation
        control = self.Kp * err + self.Kd * err_dot + self.Ki * self.err_int

        return control

    def reset(self):
        """
        Resets the controller to its initial state

        """
        super(CPIDController, self).reset()
        self.err_int = []


class CPotentialFieldController(CController):
    """Controller based in potential fields approach. Depends on pybullet for distance checks from the robot
    model to the obstacles

    Attributes:
        Krep (float): Gain for the repulsive action that pushes the state aways from obstacles
        ctrl (CController): Controller used to provide the attractive action that pushes the state towards the reference
        model (int): Pybullet unique ID for the robot model
        eef_link (int): End effector link index
        obstacles (list): Pybullet unique ID for obstacles
        max_dist (float): Limit distance to objects used to compute a repulsive action. Objects further will not be used
        sim_id (int): Pybullet simulator instance id
    """
    def __init__(self, Krep, ctrl):
        """
        :param Krep: Gain applied to the repulsive control action
        :param ctrl: Controller used to compute the attractive control action
        """
        super(CPotentialFieldController, self).__init__()
        self.ctrl = ctrl
        self.Krep = Krep
        self.model = 0
        self.eef_link = 0
        self.obstacles = []
        self.max_dist = 0.2
        self.sim_id = 0

    def set_model(self, model, eef_link, physicsClientId=0):
        """
        :param model: (int) Pybullet unique ID for the robot model
        :param eef_link: (int) End effector link index
        :param physicsClientId: (int) Pybullet simulator instance id
        """
        self.model = model
        self.eef_link = eef_link
        self.sim_id = physicsClientId

    def set_obstacles(self, obstacles, max_dist=0.5):
        """
        :param obstacles: (list) Pybullet unique ID for obstacles
        :param max_dist: (float) Objects further than max_dist will not be used for repulsive action computation
        """
        self.obstacles = obstacles
        self.max_dist = max_dist

    def get_command(self, state, ref):
        """Obtain the control action for the provided current and desired states taking into account the obstacle list
        controlled object model and end effector link (must be provided through previous set_model and set_obstacles)

        Control action computed as a weighted sum of the repulsive and attractive controllers
            $u = K_rep * u_r + u_a$

        Where:
            d_i = vector representing the closest distance from the end effector to an obstacle
            $u_r = \sum_{i=0}^n \hat{d_i} \frac{1}{|d_i|^3}$
            u_a = self.ctrl(ref,state)

        :param state: current state
        :param ref: desired state
        :return: control action computed as: $u = K_rep u_r + u_a$
        """

        # Obtain closest points from the end effector to the obstacles
        points = []
        for body in self.obstacles:
            closest_points = pybullet.getClosestPoints(bodyA=self.model, bodyB=body, distance=self.max_dist,
                                                       linkIndexA=self.eef_link, linkIndexB=-1, physicsClientId=self.sim_id)
            if len(closest_points) > 0:
                points.extend(closest_points)

        # Compute a repulsive action
        cmd_rep = np.zeros(len(state))
        for cpoint in points:
            cdir = np.array(cpoint[6]) - np.array(cpoint[5])
            cdir = cdir / np.linalg.norm(cdir)
            dist = cpoint[8]
            cmd_rep = cmd_rep + cdir * (1/(dist*dist))

        cmd_pid = self.ctrl.get_command(state, ref)
        self.err = self.ctrl.err

        return cmd_pid + self.Krep * cmd_rep * np.array([1,1,0.1])

    def reset(self):
        """
        Resets the controller to its initial state

        """
        super(CPotentialFieldController, self).reset()
        self.ctrl.reset()
