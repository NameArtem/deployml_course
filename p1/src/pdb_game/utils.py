class UnnecessaryError(Exception):
    pass


def i_just_throw_an_exception():
    value = 1
    def some_inner_function():
        value += 1

    some_value = "Я не знаю, что вы ожидали"
    raise UnnecessaryError("Ошибка выбора...")