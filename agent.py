import numpy as np
import math
from game import Board, UP, RIGHT, DOWN, LEFT, action_name
from game import IllegalAction, GameOver
from queue import Queue
import torch
import torch.nn as nn
import torch.optim as optim
from policy_gradient.actor_critic import ActorCritic
from policy_gradient.roll_out_storage import RolloutStorage
class nTupleNetwork:
    def __init__(self, tuples, symmetric_sampling=True, after_state=True, lambd= 0.5):
        self.TUPLES = tuples
        self.m = len(tuples)
        self.TARGET_PO2 = 15
        self.lambd = 0.1
        self.h = int(-1/math.log(lambd,10))
        self.symmetric_sampling = symmetric_sampling
        self.after_state = after_state
        self.LUTS = self.initialize_LUTS(self.TUPLES)


    def initialize_LUTS(self, tuples):
        LUTS = []
        for tp in tuples:
            if self.symmetric_sampling:
                LUTS.append(np.zeros((self.TARGET_PO2 + 1) ** len(tp[0])))
            else:
                LUTS.append(np.zeros((self.TARGET_PO2 + 1) ** len(tp)))
        return LUTS

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            if v >= self.TARGET_PO2:
                raise ValueError(
                    "digit %d should be smaller than the base %d" % (v, self.TARGET_PO2)
                )
            n += v * k
            k *= self.TARGET_PO2
        return n

    def V(self, board, delta=None, debug=False):
        """Return the expected total future rewards of the board.
        Updates the LUTs if a delta is given and return the updated value.
        """
        if debug:
            print(f"V({board})")
        vals = []
        if self.symmetric_sampling:
            for i, (tp, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
                for tuple in tp:
                    tiles = [board[i] for i in tuple]
                    tpid = self.tuple_id(tiles)
                    if delta is not None:
                        LUT[tpid] += delta
                        
                    v = LUT[tpid]
                    if debug:
                        print("board: ",board)
                        print("tp: ",tp)
                        print("tuple: ",tuple)
                        print("tiles: ",tiles)
                        print("tpid: ",tpid)
                        print("LUT[tpid]: ",LUT[tpid])
                        print(f"LUTS[{i}][{tiles}]={v}")
                    vals.append(v)
            return np.mean(vals)
        else:
            for i, (tp, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
                tiles = [board[i] for i in tp]
                tpid = self.tuple_id(tiles)
                if delta is not None:
                    LUT[tpid] += delta
                v = LUT[tpid]
                if debug:
                    print(f"LUTS[{i}][{tiles}]={v}")
                vals.append(v)
            return np.mean(vals)
    def evaluate(self, s, a):
        "Return expected total rewards of performing action (a) on the given board state (s)"
        b = Board(s)
        try:
            r = b.act(a)
            s_after = b.copyboard()
            if ~(self.after_state):
                b.spawn_tile(random_tile=True)
                s_next = b.copyboard()
        except(IllegalAction, GameOver) as e:
            return 0
        if self.after_state:
            return r + self.V(s_after)
        else:
            return r + self.V(s_next)
    

    def best_action(self, s):
        "returns the action with the highest expected total rewards on the state (s)"
        a_best = None
        r_best = -1
        for a in [UP, RIGHT, DOWN, LEFT]:
            r = self.evaluate(s, a)
            if r > r_best:
                r_best = r
                a_best = a
        return a_best
    
    def TDupdate(self, history, current_step, delta, alpha=0.1,lambd=0.5):
        T = min(self.h+1,current_step+1)
        for i in range(T):
            self.V(history[current_step-i].s_after,alpha/self.m*delta*(lambd**i))

    def TCupdate(self, step, delta, beta=1,lambd=0.5):
        T = min(self.h+1,step)
        for i in range(T):
            after_state = self.history[T-1-i]
            self.actualTCupdate(delta,after_state, beta=beta, lambd=lambd)
            # alpha = abs(self.E[after_state])/self.A[after_state]
            # self.V(after_state,beta*alpha/self.m*delta*(lambd**i))
            # self.E[after_state] += delta
            # self.A[after_state] += abs(delta) 

    def actualTCupdate(self, diff,after_state,beta=1.0,lambd=0.5):
        alpha = abs(self.E[after_state])/self.A[after_state]
        self.V(after_state,beta*alpha/self.m*diff)
        self.E[after_state] += diff
        self.A[after_state] += abs(diff)    

    def delayedTCupdate(self, history, his_len, beta=1.0, lambd=0.5):
        if his_len > self.h:
            diff = 0
            for i in range(self.h):
                diff += history[-self.h-1].delta*(lambd**i)
            self.actualTCupdate(diff, history[-1].s_after,beta=beta,lambd=lambd)

    def final(self,history,his_len,beta=1.0,lambd=0.5):
        for i in reversed(range(self.h)):
            diff=0
            for j in range(i):
                diff+= history[-i+j]*(lambd**j)
            self.actualTCupdate(diff, history[-i],diff)


    def GetDelta(self, s_after, s_next, debug=False):
        """Learn from a transition experience by updating the belief
        on the after state (s_after) towards the sum of the next transition rewards (r_next) and
        the belief on the next after state (s_after_next).

        """
        a_next = self.best_action(s_next)
        b = Board(s_next)
        try:
            r_next = b.act(a_next)
            s_after_next = b.copyboard()
            if self.after_state:
                v_after_next = self.V(s_after_next)
            else:
                b.spawn_tile(random_tile=True)
                s_next_next = b.copyboard()
                v_next_next = self.V(s_next_next)

        except (IllegalAction, GameOver) as e:
            r_next = 0
            if self.after_state:
                v_after_next = 0
            else:
                v_next_next = 0
        if self.after_state:
            delta = r_next + v_after_next - self.V(s_after)
        else:
            delta = r_next + v_next_next - self.V(s_next)

        if debug:
            print("s_next")
            Board(s_next).display()
            print("a_next", action_name(a_next), "r_next", r_next)
            print("s_after_next")
            Board(s_after_next).display()
            self.V(s_after_next, debug=True)
            print(
                f"delta ({delta:.2f}) = r_next ({r_next:.2f}) + v_after_next ({v_after_next:.2f}) - V(s_after) ({V(s_after):.2f})"
            )
            print(
                f"V(s_after) <- V(s_after) ({V(s_after):.2f}) + alpha * delta ({alpha} * {delta:.1f})"
            )

        return delta

    def update(self,transition_history, delta, current_step, mode, alpha=0.1, beta=1.0, lambd=0.5):
        if mode=='TD0':
            if self.after_state:
                self.V(transition_history[current_step].s_after, alpha*delta)
            else:
                self.V(transition_history[current_step].s_next, alpha*delta)

        if mode=='TDlambda':
            self.TDupdate(transition_history, current_step, delta,alpha=alpha,lambd=lambd)
        if mode =='TClambda':
            self.TCupdate(transition_history, current_step, delta,beta=beta,lambd=lambd)
        if mode =='delayedTClambda':
            self.delayedTCupdate(transition_history,current_step,beta=beta,lambd=lambd)
    
    def termination_update(self, transition_history, his_len, mode, alpha=0.1,beta=1.0,lambd=0.5):
        if mode =='TDlambda' and his_len!=0:
            self.TDupdate(transition_history,his_len,-self.V(transition_history[-1].s_after),alpha,lambd)
        # if mode =='delayedTClambda' and his_len!=0:
        #     self.final(transition_history,his_len,delta,beta,lambd)


class PolicyGradient:
    actor_critic: ActorCritic
    def __init__(self,
                 tuples,
                 actor_critic,
                 agent='ActorCritic',
                 symmetric_sampling=True, 
                 after_state=True,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):
        self.TUPLES = tuples
        self.m = len(tuples)
        self.TARGET_PO2 = 15
        self.lambd = 0.5
        self.h = 3
        self.agent = agent
        self.symmetric_sampling = symmetric_sampling
        self.after_state = after_state
        self.LUTS = self.initialize_LUTS(self.TUPLES)
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def initialize_LUTS(self, tuples):
        LUTS = []
        for tp in tuples:
            if self.symmetric_sampling:
                LUTS.append(np.zeros((self.TARGET_PO2 + 1) ** len(tp[0])))
            else:
                LUTS.append(np.zeros((self.TARGET_PO2 + 1) ** len(tp)))
        return LUTS

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            if v >= self.TARGET_PO2:
                raise ValueError(
                    "digit %d should be smaller than the base %d" % (v, self.TARGET_PO2)
                )
            n += v * k
            k *= self.TARGET_PO2
        return n

    def V(self, board, delta=None, debug=False):
        """Return the expected total future rewards of the board.
        Updates the LUTs if a delta is given and return the updated value.
        """
        if debug:
            print(f"V({board})")
        vals = []
        if self.symmetric_sampling:
            for i, (tp, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
                for tuple in tp:
                    tiles = [board[i] for i in tuple]
                    tpid = self.tuple_id(tiles)
                    if delta is not None:
                        LUT[tpid] += delta
                        
                    v = LUT[tpid]
                    if debug:
                        print("board: ",board)
                        print("tp: ",tp)
                        print("tuple: ",tuple)
                        print("tiles: ",tiles)
                        print("tpid: ",tpid)
                        print("LUT[tpid]: ",LUT[tpid])
                        print(f"LUTS[{i}][{tiles}]={v}")
                    vals.append(v)
            return np.mean(vals)
        else:
            for i, (tp, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
                tiles = [board[i] for i in tp]
                tpid = self.tuple_id(tiles)
                if delta is not None:
                    LUT[tpid] += delta
                v = LUT[tpid]
                if debug:
                    print(f"LUTS[{i}][{tiles}]={v}")
                vals.append(v)
            return np.mean(vals)
    def evaluate(self, s, a):
        "Return expected total rewards of performing action (a) on the given board state (s)"
        b = Board(s)
        try:
            r = b.act(a)
            s_after = b.copyboard()
            if ~(self.after_state):
                b.spawn_tile(random_tile=True)
                s_next = b.copyboard()
        except(IllegalAction, GameOver) as e:
            return 0
        if self.after_state:
            return r + self.V(s_after)
        else:
            return r + self.V(s_next)
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones) # not implemented!
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss