# The camera reports residual beacon error: target LOS minus actual FSM correction.
# The *capture* tick is different from the later delivery tick.
uint64 capture_tick
float64 error_x
float64 error_y
float64 fsm_x_at_capture
float64 fsm_y_at_capture
bool detected
