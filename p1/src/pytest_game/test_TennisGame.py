import TennisGame

def test_zero_zero_score():
    # Arrange
    game = TennisGame.TennisGame()
    
    # Act
    score = game.score()
    
    # Asert
    assert score == "0 - 0"
    
def test_server_scores_a_point():

    # Arrange
    game = TennisGame.TennisGame()
    originalServerPoints = game.serverPoints
    
    # Act
    game.serverScoresPoint()
    
    # Assert
    assert game.serverPoints == originalServerPoints + 1
    
def test_15_zero_score():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()
    
    # Act
    score = game.score()
    
    # Assert
    assert score == "15 - 0"

def test_receiver_scores_a_point():
    # Arrange
    game = TennisGame.TennisGame()
    originalReceiverPoints = game.receiverPoints
    
    # Act
    game.receiverScoresPoint()
    
    # Assert
    assert game.receiverPoints == originalReceiverPoints + 1
    
def test_zero_15_score():
    # Arrange
    game = TennisGame.TennisGame()
    game.receiverScoresPoint()
    
    # Act
    score = game.score()
    
    # Assert
    assert score == "0 - 15"

def test_30_30_score():
    # Arrange
    game = TennisGame.TennisGame()
    game.receiverScoresPoint()
    game.receiverScoresPoint()
    game.serverScoresPoint()
    game.serverScoresPoint()
    
    # Act
    score = game.score()
    
    # Assert
    assert score == "30 - 30"

def test_45_30_score():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()
    game.serverScoresPoint()
    game.serverScoresPoint()
    game.receiverScoresPoint()
    game.receiverScoresPoint()
    
    # Act
    score = game.score()
    
    # Assert
    assert score == "40 - 30"

def test_30_45_score():
    # Arrange
    game = TennisGame.TennisGame()
    game.receiverScoresPoint()
    game.receiverScoresPoint()
    game.receiverScoresPoint()
    game.serverScoresPoint()
    game.serverScoresPoint()
    
    # Act
    score = game.score()
    
    # Assert
    assert score == "30 - 40"

def test_deuce_score():
    # Arrange
    game = TennisGame.TennisGame()
    game.receiverScoresPoint()  # 0 - 15
    game.receiverScoresPoint()  # 0 - 30
    game.serverScoresPoint()    # 15 - 30
    game.serverScoresPoint()    # 30 - 30
    game.receiverScoresPoint()  # 40 - 30
    game.serverScoresPoint()    # 40 - 40 or Deuce
   
    # Act
    score = game.score()
    
    # Assert
    assert score == "Ничья"
    
def test_game_server_before_deuce():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()
    game.serverScoresPoint()
    game.serverScoresPoint()
    game.serverScoresPoint()
    game.receiverScoresPoint()
    game.receiverScoresPoint()

    # Act
    score = game.score()
    
    # Assert
    assert score == "Игра на стороне 1"
    
def test_game_server_after_deuce():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()    # 15 - 0
    game.serverScoresPoint()    # 30 - 0
    game.serverScoresPoint()    # 40 - 0
    game.receiverScoresPoint()  # 40 - 15
    game.receiverScoresPoint()  # 40 - 30
    game.receiverScoresPoint()  # Deuce
    game.serverScoresPoint()    # Advantage 1
    game.serverScoresPoint()    # Game 1

    # Act
    score = game.score()
    
    # Assert
    assert score == "Игра на стороне 1"
    
def test_game_receiver_before_deuce():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()    # 15 - 0
    game.serverScoresPoint()    # 30 - 0
    game.receiverScoresPoint()  # 30 - 15
    game.receiverScoresPoint()  # 30 - 30
    game.receiverScoresPoint()  # 30 - 40
    game.receiverScoresPoint()  # Game 2

    # Act
    score = game.score()
    
    # Assert
    assert score == "Игра на стороне 2"

def test_game_receiver_after_deuce():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()    # 15 - 0
    game.serverScoresPoint()    # 30 - 0
    game.receiverScoresPoint()  # 30 - 15
    game.receiverScoresPoint()  # 30 - 30
    game.receiverScoresPoint()  # 30 - 40
    game.serverScoresPoint()    # Deuce
    game.receiverScoresPoint()  # Advantage 2
    game.receiverScoresPoint()  # Game 2

    # Act
    score = game.score()
    
    # Assert
    assert score == "Игра на стороне 2"

def test_advantage_server():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()    # 15 - 0
    game.serverScoresPoint()    # 30 - 0
    game.serverScoresPoint()    # 40 - 0
    game.receiverScoresPoint()  # 40 - 15
    game.receiverScoresPoint()  # 40 - 30
    game.receiverScoresPoint()  # Deuce
    game.serverScoresPoint()    # Advantage 1
    game.receiverScoresPoint()  # Deuce
    game.serverScoresPoint()    # Advantage 1
    
    # Act
    score = game.score()
    
    # Assert
    assert score == "Преимущество на стороне 1"
    
def test_advantage_receiver():
    # Arrange
    game = TennisGame.TennisGame()
    game.serverScoresPoint()    # 15 - 0
    game.serverScoresPoint()    # 30 - 0
    game.receiverScoresPoint()  # 30 - 15
    game.receiverScoresPoint()  # 30 - 30
    game.receiverScoresPoint()  # 30 - 40
    game.serverScoresPoint()    # Deuce
    game.receiverScoresPoint()  # Advantage 2
    game.serverScoresPoint()    # Deuce
    game.receiverScoresPoint()  # Advantage 2

    # Act
    score = game.score()
    
    # Assert
    assert score == "Преимущество на стороне 2"
