# Baggable record of the Blueboat thrust-to-PWM mapping and MAVLink RC override.
std_msgs/Header header
builtin_interfaces/Time source_command_stamp
builtin_interfaces/Time received_stamp
float64 port_thrust_newtons
float64 starboard_thrust_newtons
uint16 port_pwm
uint16 starboard_pwm
uint16 throttle_pwm
uint16 steering_pwm
bool live_send
bool stale
string source

