from math import log2

def AURAC_v1(accs, ranks):
    assert len(accs) == len(ranks)
    n = len(accs)
    if n == 1:
        return accs[0]
    return sum([
        (ranks[i+1] - ranks[i]) * (accs[i] + accs[i+1]) / 2
        for i in range(n-1)
    ]) / (ranks[-1] - ranks[0])

def AURAC_v2(accs, ranks):
    assert len(accs) == len(ranks)
    n = len(accs)
    if n == 1:
        return accs[0]
    return sum([
        (log2(ranks[i+1]) - log2(ranks[i])) * (accs[i] + accs[i+1]) / 2
        for i in range(n-1)
    ]) / (log2(ranks[-1]) - log2(ranks[0]))