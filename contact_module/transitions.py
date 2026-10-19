"""Finite-radio experience boundaries, from one own decision to the next.

Every action services all senders. A sender's delivery reward therefore spans
other robots' slots as well as its own. This collector closes that interval at
the next actual observation, without changing actions or the SAC algorithm.
"""


class DecisionTransitions:
    def __init__(self, contact, robots):
        self.contact = contact
        self.robots = robots
        self.pending = [None] * len(robots)
        self.initial_delivery_cells = 0.
        self.settled_delivery_cells = 0.

    def _delivery(self, robot):
        cells = self.contact.delivery_cells[robot] - self.contact.claimed_delivery_cells[robot]
        if cells < 0:
            raise RuntimeError('Delivery ledger moved backwards')
        self.contact.claimed_delivery_cells[robot] += cells
        return float(cells)

    def begin(self, robot, observation):
        if self.pending[robot] is None:
            # No action exists before the first decision. Keep these genuine
            # initial-condition events explicit rather than crediting a future
            # action for past deliveries. Usually zero at the common start.
            self.initial_delivery_cells += self._delivery(robot)
        else:
            self._finish(robot, observation, False)

    def action_reward(self, robot, reward):
        if self.pending[robot] is not None:
            raise RuntimeError('Previous decision transition was not finalized')
        # Env's local exploration/distance/time/shaping terms are unchanged.
        # Replace its own-slot delivery term with all actual sender deliveries
        # in this decision interval, once, when the interval closes.
        self.pending[robot] = float(reward - self.contact.last_delivery_reward)

    def team_reward(self, reward):
        if any(value is None for value in self.pending):
            raise RuntimeError('Missing action at team reward boundary')
        self.pending = [value + reward for value in self.pending]

    def _finish(self, robot, observation, done):
        if self.pending[robot] is None:
            raise RuntimeError('Cannot finalize a transition without an action')
        cells = self._delivery(robot)
        self.settled_delivery_cells += cells
        self.robots[robot].save_next_observations(observation)
        self.robots[robot].save_reward_done(self.pending[robot] + cells / 1000., done)
        self.pending[robot] = None

    def finish(self, observations, done):
        for robot, observation in enumerate(observations):
            self._finish(robot, observation, done)
        residual = (self.contact.radio.unique_cells - self.settled_delivery_cells
                    - self.initial_delivery_cells)
        if abs(residual) > 1e-8:
            raise RuntimeError('Unsettled actual-delivery reward: {}'.format(residual))
        for robot in self.robots:
            if len({len(column) for column in robot.episode_buffer}) != 1:
                raise RuntimeError('Incomplete finite decision transition')
        self.contact.own_slot_delivery_reward = self.contact.reward_sums['delivery']
        self.contact.reward_sums['delivery'] = self.settled_delivery_cells / 1000.
        self.contact.initial_delivery_cells = self.initial_delivery_cells
        self.contact.delivery_ledger_residual = residual
