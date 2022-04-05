test = {"a": 1, "b": 1.1, "c": 2, "d": 2.1, "e": 1.9, "f": 0.9, "g": 3, "h": 4, "i": 5}

result = dict()


def check_list(check_dict: dict, check_value: float, std: float) -> bool:
    count = 0
    values = check_dict.values()
    if values:
        for value in values:
            if abs(check_value - value) < std:
                count += 1
    return True if count > 0 else False


for key, value in test.items():
    if not check_list(result, value, 0.2):
        result[key] = value
