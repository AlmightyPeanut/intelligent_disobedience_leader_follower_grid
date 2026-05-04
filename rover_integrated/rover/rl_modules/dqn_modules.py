import torch
import tree
from ray.rllib.algorithms.dqn.default_dqn_rl_module import QF_PREDS, QF_NEXT_PREDS, QF_TARGET_NEXT_PREDS, ATOMS, \
    QF_LOGITS, QF_PROBS, QF_TARGET_NEXT_PROBS
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorStructType, TensorType


class LearnedValidatorDQN(DefaultDQNTorchRLModule):
    @override(RLModule)
    def _forward_train(
            self, batch: dict[str, TensorType]
    ) -> dict[str, TensorStructType]:
        if self.inference_only:
            raise RuntimeError(
                "Trying to train a module that is not a learner module. Set the "
                "flag `inference_only=False` when building the module."
            )
        output = {}

        # If we use a double-Q setup.
        if self.uses_double_q:
            # Then we need to make a single forward pass with both,
            # current and next observations.
            if isinstance(batch[Columns.OBS], dict):
                batch_base = {
                    Columns.OBS: tree.map_structure(
                        lambda obs, next_obs: torch.concat(
                            [obs, next_obs], dim=0
                        ),
                        batch[Columns.OBS],
                        batch[Columns.NEXT_OBS],
                    )
                }
            else:
                batch_base = {
                    Columns.OBS: torch.concat(
                        [batch[Columns.OBS], batch[Columns.NEXT_OBS]], dim=0
                    ),
                }
            # If this is a stateful module add the input states.
            if Columns.STATE_IN in batch:
                # Add both, the input state for the actual observation and
                # the one for the next observation.
                batch_base.update(
                    {
                        Columns.STATE_IN: tree.map_structure(
                            lambda t1, t2: torch.cat([t1, t2], dim=0),
                            batch[Columns.STATE_IN],
                            batch[Columns.NEXT_STATE_IN],
                        )
                    }
                )
        # Otherwise we can just use the current observations.
        else:
            batch_base = {Columns.OBS: batch[Columns.OBS]}
            # If this is a stateful module add the input state.
            if Columns.STATE_IN in batch:
                batch_base.update({Columns.STATE_IN: batch[Columns.STATE_IN]})

        batch_target = {Columns.OBS: batch[Columns.NEXT_OBS]}

        # If we have a stateful encoder, add the states for the target forward
        # pass.
        if Columns.NEXT_STATE_IN in batch:
            batch_target.update({Columns.STATE_IN: batch[Columns.NEXT_STATE_IN]})

        # Q-network forward passes.
        qf_outs = self.compute_q_values(batch_base)
        if self.uses_double_q:
            output[QF_PREDS], output[QF_NEXT_PREDS] = torch.chunk(
                qf_outs[QF_PREDS], chunks=2, dim=0
            )
        else:
            output[QF_PREDS] = qf_outs[QF_PREDS]
        # The target Q-values for the next observations.
        qf_target_next_outs = self.forward_target(batch_target)
        output[QF_TARGET_NEXT_PREDS] = qf_target_next_outs[QF_PREDS]
        # We are learning a Q-value distribution.
        if self.num_atoms > 1:
            # Add distribution artefacts to the output.
            # Distribution support.
            output[ATOMS] = qf_target_next_outs[ATOMS]
            # Original logits from the Q-head.
            output[QF_LOGITS] = qf_outs[QF_LOGITS]
            # Probabilities of the Q-value distribution of the current state.
            output[QF_PROBS] = qf_outs[QF_PROBS]
            # Probabilities of the target Q-value distribution of the next state.
            output[QF_TARGET_NEXT_PROBS] = qf_target_next_outs[QF_PROBS]

        # Add the states to the output, if the module is stateful.
        if Columns.STATE_OUT in qf_outs:
            output[Columns.STATE_OUT] = qf_outs[Columns.STATE_OUT]
        # For correctness, also add the output states from the target forward pass.
        # Note, we do not backpropagate through this state.
        if Columns.STATE_OUT in qf_target_next_outs:
            output[Columns.NEXT_STATE_OUT] = qf_target_next_outs[Columns.STATE_OUT]

        return output