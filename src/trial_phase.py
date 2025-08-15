from .trial_state import TrialState


class TrialPhase:
    __slots__ = (
        "current_phase",
        "first_phase",
        "last_phase",
        "thresholds",
    )

    def __init__(
        self,
        stop_iteration: int,
        iteration_per_phase: int,
    ) -> None:
        self.current_phase = 0
        self.first_phase = 0
        self.last_phase = max(
            stop_iteration // iteration_per_phase
            - (
                1
                if stop_iteration % iteration_per_phase < iteration_per_phase // 2
                else 0
            ),
            0,
        )
        self.thresholds = [
            i * iteration_per_phase
            for i in range(self.first_phase, self.last_phase + 1)
        ]

    def update_phase(self, trial_states: list[TrialState]) -> None:
        if self.current_phase == self.last_phase:
            return

        count = 0

        for trial in trial_states:
            if trial.phase == self.current_phase + 1:
                count += 1
        if count > len(trial_states) - 20:
            self.current_phase += 1

    def get_trial_phase(self, trial_state: TrialState) -> int:
        if trial_state.phase == self.last_phase:
            return self.last_phase

        if trial_state.iteration >= self.thresholds[trial_state.phase + 1]:
            return trial_state.phase + 1

        return trial_state.phase

    def is_trial_exceeding(self, trial_state: TrialState) -> bool:
        if self.current_phase == self.last_phase:
            return False

        return trial_state.phase > self.current_phase
