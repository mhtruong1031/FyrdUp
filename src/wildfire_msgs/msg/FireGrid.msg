# Fire grid state message
# Represents a 2D grid of fire intensities

std_msgs/Header header

# Grid dimensions
uint32 width
uint32 height

# Grid cell size in meters
float32 cell_size

# Fire intensity values (0.0 = no fire, 1.0 = maximum intensity)
# Stored as 1D array, access via: intensity[y * width + x]
float32[] intensity
