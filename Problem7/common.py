def diff_func(history):
    diff = []
    for i in range(len(history)-1):
        diff.append(history[i] - history[-1])
    return diff
