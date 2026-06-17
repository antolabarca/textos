def validate_distribution(dist):

    total = sum(dist.values())

    if not all(0 <= v <= 1 for v in dist.values()):
        return False

    if abs(total - 1.0) > 0.01:
        return False

    return True