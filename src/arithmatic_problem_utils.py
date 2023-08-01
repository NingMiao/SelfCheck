def get_exact_answer(input_text):
    try:
        answer_with_unit = input_text[:-1].split('The answer is ')[1]
    except:
        answer_with_unit = input_text[-50:-1]
    answer_number_only = ''.join(filter(str.isdigit, answer_with_unit))
    return answer_number_only