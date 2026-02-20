from __future__ import annotations

import numpy as np

from . import BaseController


class Controller(BaseController):
  """
  Feedforward + feedback controller.

  Feedforward is a linear inverse model fit from labeled pre-control samples.
  Feedback stabilizes tracking under rollout mismatch and model uncertainty.
  """

  def __init__(self):
    # Linear inverse feedforward coefficients:
    # [target_t, target_t+1, target_t+2, roll_lataccel, v_ego, a_ego, bias]
    self.ff_w = np.array(
      [-0.04867, 0.11253, 0.30386, -0.64179, -0.00419, -0.01417, 0.06060],
      dtype=np.float64,
    )

    # Feedback terms (starting seed, tuned by rollout experiments).
    self.p = 0.24456
    self.i = 0.08028
    self.d = 0.03273
    self.lookahead = 0.75213
    self.decay = 0.99738
    self.int_clip = 99.61319
    self.da_max = 0.48561
    self.act_alpha = 0.31384

    self.err_i = 0.0
    self.prev_err = 0.0
    self.prev_act = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    f1 = float(future_plan.lataccel[0]) if len(future_plan.lataccel) > 0 else float(target_lataccel)
    f2 = float(future_plan.lataccel[1]) if len(future_plan.lataccel) > 1 else f1

    ff = float(
      self.ff_w[0] * float(target_lataccel)
      + self.ff_w[1] * f1
      + self.ff_w[2] * f2
      + self.ff_w[3] * float(state.roll_lataccel)
      + self.ff_w[4] * float(state.v_ego)
      + self.ff_w[5] * float(state.a_ego)
      + self.ff_w[6]
    )

    target_blend = ((1.0 - self.lookahead) * float(target_lataccel)) + (self.lookahead * f1)
    err = target_blend - float(current_lataccel)
    self.err_i = float(np.clip((self.err_i * self.decay) + err, -self.int_clip, self.int_clip))
    derr = err - self.prev_err
    self.prev_err = err

    raw = ff + (self.p * err) + (self.i * self.err_i) + (self.d * derr)
    slew_limited = self.prev_act + float(np.clip(raw - self.prev_act, -self.da_max, self.da_max))
    act = (self.act_alpha * self.prev_act) + ((1.0 - self.act_alpha) * slew_limited)
    self.prev_act = float(act)
    return float(act)
