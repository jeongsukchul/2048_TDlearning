import numpy as np
import math
from game import Board, UP, RIGHT, DOWN, LEFT, action_name
from game import IllegalAction, GameOver
from queue import Queue

class nTupleNetwork:
    def __init__(self, tuples, symmetric_sampling=True):
        self.TUPLES = tuples
        self.m = len(tuples)
        self.TARGET_PO2 = 15
        self.lambd = 0.5
        self.h = 3
        self.LUTS = self.initialize_LUTS(self.TUPLES)
        self.E = np.zeros(self.m)
        self.A = np.zeros(self.m)
        self.symmetric_sampling = symmetric_sampling
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
        except IllegalAction:
            return 0
        return r + self.V(s_after)

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
            v_after_next = self.V(s_after_next)
        except IllegalAction:
            r_next = 0
            v_after_next = 0

        delta = r_next + v_after_next - self.V(s_after)
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
            self.V(transition_history[current_step].s_after, alpha*delta)
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