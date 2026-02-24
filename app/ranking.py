from datetime import datetime
from math import log
from dateutil.parser import parse


def compute_recency_score(created_at):
    days = (datetime.now() - parse(created_at)).days
    return 1 / (days + 1)


def compute_usage_score(usage_count):
    return log(usage_count + 1)


def final_score(semantic, keyword, recency, usage):
    return (
        0.6 * semantic +
        0.2 * keyword +
        0.1 * recency +
        0.1 * usage
    )