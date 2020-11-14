
class TennisGame:
    pointNames = ["0", "15", "30", "40"]

    def __init__(self):
        self.serverPoints = 0
        self.receiverPoints = 0
        
    def score(self):
        if ((self.serverPoints < 3) and (self.receiverPoints < 3) or \
            (self.serverPoints == 3) and (self.receiverPoints < 3) or \
            (self.serverPoints < 3) and (self.receiverPoints == 3)):
            return TennisGame.pointNames[self.serverPoints] + \
                " - " + \
                TennisGame.pointNames[self.receiverPoints]
            
        if (self.serverPoints == self.receiverPoints):
            return "Ничья"
            
        if ((3 < self.serverPoints) or (3 < self.receiverPoints)) and \
           (1 < abs(self.serverPoints - self.receiverPoints)):
            if (self.serverPoints > self.receiverPoints):
                return "Игра на стороне 1"
            else:
                return "Игра на стороне 2"
        
        if ((3 <= self.serverPoints) and (3 <= self.receiverPoints)):
            if (self.serverPoints > self.receiverPoints):
                return "Преимущество на стороне 1"
            else:
                return "Преимущество на стороне 2"
        
        return "Не доступный вариант!"

    def serverScoresPoint(self):
        self.serverPoints += 1
        
    def receiverScoresPoint(self):
        self.receiverPoints += 1



if __name__ == "__main__":

    import random

    game = TennisGame()

    # Act
    score = game.score()
    print(score)

    for _ in range(0, 10):

        if random.choice([0,1]) == 0:
            game.serverScoresPoint()
        else:
            game.receiverScoresPoint()

        score = game.score()
        print(score)

        if score == 'Ничья' \
                or 'Преимущество' in score \
                or 'Игра на' in score:
            break
