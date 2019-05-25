## Project Task introduction
This task involves reinforcement learning and will no longer use the tags in task 1. To simplify the problem, this task sets that each tick has at most 5 hand long positions and 5 hand short positions. Long positions and short positions cannot be held at the same time. A tick can only have one action at a time. Positions can be increased or decreased (with unit equals one hand) through buying and selling, and the absolute value of change in the number of positions of one action cannot exceed one hand. The current state can be maintained by an idle action. When the buying action is executed, the purchase will be successful and will not have any impact on the market. The price is AskPrice1 of the current tick. When the selling action is executed, the sell will be successful and will have no eﬀect on the market. The price is BidPrice1 of the current tick. Finally, you should include in your report: the number of buying and spelling on testing set, the average price to buy and the average price to sell. Besides, attach action selection for each tick on testing set for submission. 

## code reference
[vin](https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks)

[tcn](https://github.com/Songweiping/TCN-TF)