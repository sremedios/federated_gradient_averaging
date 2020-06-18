def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n