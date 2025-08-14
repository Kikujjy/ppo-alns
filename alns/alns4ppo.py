from typing import Dict, List, Optional, Protocol, Tuple
import numpy.random as rnd
from alns.State import State


class _OperatorType(Protocol):
    __name__: str

    def __call__(
        self,
        state: State,
        rng: rnd.Generator,
        **kwargs,
    ) -> State: ...  # pragma: no cover


class ALNS4PPO:
    def __init__(self, rng: rnd.Generator = rnd.default_rng()):
        self._rng = rng

        self._d_ops: Dict[str, _OperatorType] = {}
        self._r_ops: Dict[str, _OperatorType] = {}

    @property
    def destroy_operators(self) -> List[Tuple[str, _OperatorType]]:
        """
        Returns the destroy operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._d_ops.items())

    @property
    def repair_operators(self) -> List[Tuple[str, _OperatorType]]:
        """
        Returns the repair operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._r_ops.items())

    def add_destroy_operator(self, op: _OperatorType, name: Optional[str] = None):
        """
        Adds a destroy operator to the heuristic instance.

        .. warning::

            A destroy operator will receive the current solution state
            maintained by the ALNS instance, not a copy. Make sure to modify
            a **copy** of this state in the destroy operator, created using,
            for example, :func:`copy.copy` or :func:`copy.deepcopy`.

        Parameters
        ----------
        op
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action. Its second
            argument is the RNG passed to the ALNS instance.
        name
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._d_ops[op.__name__ if name is None else name] = op

    def add_repair_operator(self, op: _OperatorType, name: Optional[str] = None):
        """
        Adds a repair operator to the heuristic instance.

        Parameters
        ----------
        op
            An operator that, when applied to the destroyed state, returns a
            new state reflecting its implemented repair action. Its second
            argument is the RNG passed to the ALNS instance.
        name
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._r_ops[name if name else op.__name__] = op

    def iterate(
        self,
        cur_solution: State,
        pre_solution: State,
        d_idx: int,
        r_idx: int,
        accept: int,
        **kwargs,
    ) -> Tuple[State, State]:
        """
        Execute single ALNS iteration using destroy and repair operators selected by PPO policy network.

        Parameters
        ----------
        current_solution
            Current solution
        d_idx
            Destroy operator index selected by PPO
        r_idx
            Repair operator index selected by PPO
        **kwargs
            Optional keyword arguments passed to operators

        Returns
        -------
        Tuple[State, State]
            Returns a tuple containing (updated best solution, repaired solution)
        """

        if accept == 0:
            cur_solution = pre_solution
        elif accept == 1:
            d_name, d_operator = self.destroy_operators[d_idx]
            r_name, r_operator = self.repair_operators[r_idx]
            destroyed = d_operator(cur_solution, self._rng, **kwargs)
            cand = r_operator(destroyed, self._rng, **kwargs)

            pre_solution = cur_solution
            cur_solution = cand

        return pre_solution, cur_solution

    def reset_opt(self):
        """
        Reset ALNS destroy and repair operators.
        Clear destroy operator dictionary _d_ops and repair operator dictionary _r_ops.
        """
        self._d_ops.clear()
        self._r_ops.clear()
        self._rng = rnd.default_rng()
