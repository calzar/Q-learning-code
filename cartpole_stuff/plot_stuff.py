import matplotlib.pyplot as plt
score_file = open("cart_score.txt","r")
loss_file = open("cart_loss.txt",)

count = 1
frame = []
epoch = []
score = []
loss = []
for s in score_file:
    epoch.append(count)
    score.append(float(s))
    count += 1
count = 1
for l in loss_file:
    loss.append(float(l))
    frame.append(count)
    count += 1

loss_file.close()
score_file.close()
#plot score
plt.figure(1)
#plt.subplot(211)
plt.plot(epoch,score)
plt.ylabel('Score of Game')
plt.xlabel('Games Played')
#plt.subplot(212)
#plt.plot(loss)
plt.show()

plt.figure(2)
plt.plot(frame,loss)
plt.ylabel('Loss')
plt.xlabel('Action of Agent')
plt.show()