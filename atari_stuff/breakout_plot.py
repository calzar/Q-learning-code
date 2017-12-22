import matplotlib.pyplot as plt

score_file = open("Output.txt","r")
scores = []

for s in score_file:
    scores.append(s)

games = list(range(0,len(scores)))
score_file.close()

plt.figure(1)
plt.plot(games,scores)
plt.ylabel('Score')
plt.xlabel('Games Played')
plt.show()