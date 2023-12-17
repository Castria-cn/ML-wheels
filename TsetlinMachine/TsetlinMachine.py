from tqdm import tqdm
import numpy as np
import random

def clip(value, minv, maxv):
    """
    clip the value in [minv, maxv].
    """
    if value < minv:
        return minv
    if value > maxv:
        return maxv
    return value

class TsetlinAutomaton:
    """
    Represents a TsetlinAutomaton.
    """
    REWARD = 0
    INACTION = 1
    PENALTY = 2

    EXCLUDE = 3
    INCLUDE = 4
    
    def __init__(self, state_num: int=100, init_state=-1):
        assert state_num % 2 == 0
        self.N = state_num // 2
        self.state = init_state if init_state != -1 else random.randint(1, state_num)
    
    def update(self, feedback: int):
        assert feedback in [TsetlinAutomaton.REWARD, TsetlinAutomaton.INACTION, TsetlinAutomaton.PENALTY]
        if feedback == TsetlinAutomaton.PENALTY:
            if self.state <= self.N:
                self.state += 1
            else:
                self.state -= 1
        elif feedback == TsetlinAutomaton.REWARD:
            if self.state > 1 and self.state <= self.N:
                self.state -= 1
            elif self.state > self.N:
                self.state += 1
        else:
            return
    
    @property
    def action(self):
        return TsetlinAutomaton.EXCLUDE if self.state <= self.N else TsetlinAutomaton.INCLUDE
    
    def __str__(self) -> str:
        return f'TsetlinAutomaton(state={self.state}, state_num={self.N * 2})'

class Proposition:
    def __init__(self, input_dim: int):
        self.automata = [TsetlinAutomaton() for i in range(input_dim)]
    
    def __call__(self, x: np.ndarray) -> bool:
        """
        - x: input [x1, x2, ..., xo], 1-d array
        - returns: truth value calculated by the proposition
        """
        result = True
        x = x.astype(np.bool_)
        x = np.concatenate([x, ~x])
        assert x.shape[0] == len(self.automata)
        for i, literal in enumerate(x):
            if self.automata[i].action == TsetlinAutomaton.INCLUDE:
                result = (result and literal)
        
        return result

    def __str__(self):
        return f'Propostition({[automaton.action == TsetlinAutomaton.INCLUDE for automaton in self.automata]})'

class TsetlinMachine:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def inference(self, x: np.ndarray):
        pass

class BinaryTsetlinMachine(TsetlinMachine):
    def __init__(self, input_dim: int, n: int=100, s: float=2.0, T: int=10):
        """
        - input_dim: dimension of input boolean vector
        - n: total proposition number
        - s: score
        - T: margin
        """
        assert n % 2 == 0
        self.input_dim = input_dim
        self.n = n
        self.s = s
        self.T = T
        
        self.positive = [Proposition(input_dim * 2) for i in range(n // 2)]
        self.negative = [Proposition(input_dim * 2) for i in range(n // 2)]

        self.type1 = np.array(
            [[1, self.s - 1, 0],
             [1, self.s - 1, 0],
             [1, self.s - 1, 0],
             [0, 1, self.s - 1],
             [0, self.s - 1, 1],
             [0, self.s - 1, 1],
             [0, 0, 0],
             [self.s - 1, 1, 0]]
        ).reshape(2, 2, 2, 3) / self.s

        self.type2 = np.array(
            [[0, 1, 0],
             [0, 1, 0],
             [0, 0, 1],
             [0, 1, 0],
             [0, 1, 0],
             [0, 1, 0],
             [0, 0, 0],
             [0, 1, 0]], dtype=np.float32).reshape(2, 2, 2, 3)
    
    def _type1feedback(self, x: np.ndarray, p: Proposition, result: bool):
        for k in range(2 * self.input_dim):
            lk = int(x[k] if k < self.input_dim else ~x[k - self.input_dim]) # in {0, 1}
            alpha = p.automata[k].action - TsetlinAutomaton.EXCLUDE # in {0, 1}
            result = int(result) # in {0, 1}
            dist = self.type1[alpha, result, lk]
            assert (dist > 0).all, str(dist)
            feedback = np.random.choice(range(len(dist)), p=dist) # 0: reward, 1: inaction, 2: penalty
            p.automata[k].update(feedback)
    
    def _type2feedback(self, x: np.ndarray, p: Proposition, result: bool):
        for k in range(2 * self.input_dim):
            lk = int(x[k] if k < self.input_dim else ~x[k - self.input_dim]) # in {0, 1}
            alpha = p.automata[k].action - TsetlinAutomaton.EXCLUDE # in {0, 1}
            result = int(result) # in {0, 1}
            dist = self.type2[alpha, result, lk]
            assert (dist > 0).all, str(dist)
            feedback = np.random.choice(range(len(dist)), p=dist) # 0: reward, 1: inaction, 2: penalty
            p.automata[k].update(feedback)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epoch: int=10) -> None:
        """
        Fit on the given data.
        - X: np.ndarray[bool], (n, input_dim)
        - y: np.ndarray[bool], (n, )
        """
        assert X.shape[1] == self.input_dim
        
        for i in range(epoch):
            for (X_, y_) in zip(X, y):
                positive_values = [positive(X_) for positive in self.positive] # Obtain positive clauses
                negative_values = [negative(X_) for negative in self.negative] # Obtain negative clauses
                v = sum(positive_values) - sum(negative_values)
                for j in range(self.n // 2):
                    th1 = (self.T - clip(v, -self.T, self.T)) / (2 * self.T)
                    th2 = (self.T + clip(v, -self.T, self.T)) / (2 * self.T)

                    if y_: # True
                        if random.random() <= th1:
                            self._type1feedback(X_, self.positive[j], positive_values[j])
                        if random.random() <= th1:
                            self._type2feedback(X_, self.negative[j], negative_values[j])
                    else:
                        if random.random() <= th2:
                            self._type2feedback(X_, self.positive[j], positive_values[j])
                        if random.random() <= th2:
                            self._type1feedback(X_, self.negative[j], negative_values[j])

    def inference(self, x: np.ndarray, return_score=False) -> np.ndarray:
        """
        - X: np.ndarray[bool], (n, input_dim)
        - returns: np.ndarray[bool], (n, )
        """
        assert len(x.shape) == 2

        result = []
        for x_ in x:
            positive_values = [positive(x_) for positive in self.positive]
            negative_values = [negative(x_) for negative in self.negative]
            result.append(sum(positive_values) - sum(negative_values))
        if return_score:
            return np.array(result)
        return np.array(result) >= 0

class MultiClassTsetlinMachine(TsetlinMachine):
    def __init__(self, input_dim: int, cls_num: int, n: int=100, s: float=2.0, T: int=10):
        """
        - input_dim: dimension of input boolean vector
        - n: total proposition number
        - s: score
        - T: margin
        """
        assert n % 2 == 0
        self.input_dim = input_dim
        self.cls_num = cls_num
        self.n = n
        self.s = s
        self.T = T
        
        self.tm = [BinaryTsetlinMachine(input_dim, n, s, T) for i in range(cls_num)]
    
    def fit(self, X: np.ndarray, y: np.ndarray, epoch: int=10):
        """
        Fit on the given data.
        - X: np.ndarray[bool], (n, input_dim)
        - y: np.ndarray[int], (n, )
        """
        assert X.shape[1] == self.input_dim
        
        for i in tqdm(range(epoch)):
            for (X_, y_) in zip(X, y):
                self.tm[y_].fit(np.array([X_]), np.array([True]), epoch=1)
                indices = list(range(self.cls_num))
                indices.pop(y_)
                rand_tm = np.random.choice(indices)
                self.tm[rand_tm].fit(np.array([X_]), np.array([False]), epoch=1)
    
    def inference(self, x: np.ndarray) -> np.ndarray:
        """
        - X: np.ndarray[bool], (n, input_dim)
        - returns: np.ndarray[int], (n, )
        """
        assert len(x.shape) == 2

        scores = np.stack([tm.inference(x, return_score=True) for tm in self.tm], axis=0)

        return np.argmax(scores, axis=0)