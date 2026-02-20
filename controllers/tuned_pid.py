from __future__ import annotations

import numpy as np

from . import BaseController


class Controller(BaseController):
  """
  Tuned lookahead PID with integral decay + action smoothing and slew limiting.

  The control law is:
    target_blend = (1 - lookahead) * target + lookahead * future_target_1
    err = target_blend - current
    err_i = clip(err_i * decay + err, -int_clip, int_clip)
    derr = err - prev_err
    raw = p * err + i * err_i + d * derr
    slew_limited = prev_act + clip(raw - prev_act, -da_max, da_max)
    act = act_alpha * prev_act + (1 - act_alpha) * slew_limited
  """

  def __init__(self):
    # Robust defaults from local validation.
    self.p = 0.28593
    self.i = 0.10178
    self.d = 0.15422
    self.decay = 0.99845
    self.lookahead = 0.64514
    self.da_max = 0.76003
    self.int_clip = 60.34229
    self.act_alpha = 0.47312

    self.err_i = 0.0
    self.prev_err = 0.0
    self.prev_act = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    del state  # Unused by this controller.

    target_blend = float(target_lataccel)
    if len(future_plan.lataccel) > 0:
      target_blend = ((1.0 - self.lookahead) * float(target_lataccel)) + (self.lookahead * float(future_plan.lataccel[0]))

    err = target_blend - float(current_lataccel)
    self.err_i = float(np.clip((self.err_i * self.decay) + err, -self.int_clip, self.int_clip))
    derr = err - self.prev_err
    self.prev_err = err

    raw = (self.p * err) + (self.i * self.err_i) + (self.d * derr)
    slew_limited = self.prev_act + float(np.clip(raw - self.prev_act, -self.da_max, self.da_max))

    act = (self.act_alpha * self.prev_act) + ((1.0 - self.act_alpha) * slew_limited)
    self.prev_act = float(act)
    return float(act)
