# Command from scout to firefighter to move to a position

std_msgs/Header header

string target_id
geometry_msgs/Point position
int32 priority
int32[2] fire_cell  # Grid cell coordinates of the target fire
