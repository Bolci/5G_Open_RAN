

def get_print_message(message: dict) -> list[str]:
    messages = []
    for estimator_label, scores in message.items():
        messages.append(f"Estimator={estimator_label}, validation_score = {scores['valid_score']}, test_score = {scores['test_score']}")

    return messages