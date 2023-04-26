def map_value(o_value, o_min, o_max, n_min, n_max):
    return round(((o_value - o_min) / (o_max - o_min)) * (n_max - n_min) + n_min)
