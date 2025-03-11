def get_print_message(message: dict) -> list[str]:
    """
    Generates a list of formatted strings from a dictionary of scores.

    :param message: A dictionary where keys are estimator labels and values are dictionaries
                    containing 'valid_score' and 'test_score'.
    :return: A list of formatted strings with estimator labels and their corresponding scores.
    """
    messages = []
    for estimator_label, scores in message.items():
        messages.append(f"Estimator={estimator_label}, validation_score = {scores['valid_score']}, test_score = {scores['test_score']}")

    return messages
