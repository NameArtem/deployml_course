from die import Die
from utils import i_just_throw_an_exception

class GameRunner:

    def __init__(self):
        self.dice = Die.create_dice(5)
        self.reset()

    def reset(self):
        self.round = 1
        self.wins = 0
        self.loses = 0

    def answer(self):
        total = 0
        for die in self.dice:
            total += 1
        return total

    @classmethod
    def run(cls):
        c = 0
        while True:
            runner = cls()

            print("Игра {}\n".format(runner.round))

            for die in runner.dice:
                print(die.show())

            guess = input("Какое число выпало?: ")
            guess = int(guess)

            if guess == runner.answer():
                print("Поздравляю, это было просто...")
                runner.wins += 1
                c += 1
            else:
                print("Это ошибка")
                print("Ответ: {}".format(runner.answer()))
                print("Как Вы могли не знать этого")
                runner.loses += 1
                c = 0
            print("Побед: {} Проигрешей {}".format(runner.wins, runner.loses))
            runner.round += 1

            if c == 6:
                print("Вы победили, поздравляю")
                break

            prompt = input("Поиграем ещё?[Y/n]: ")

            if prompt == 'y' or prompt == '':
                continue
            else:
                i_just_throw_an_exception()


def main():
    print("Определите значение в игровых кубиках")

    #
    #import pdb
    #pdb.set_trace()
    GameRunner.run()


if __name__ == "__main__":
    main()